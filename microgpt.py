import os
import math
import random
import urllib.request


random.seed(42)

project_directory = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(project_directory, "input.txt")

# Each line is one training document: a single human name.
if not os.path.exists(input_path):
    names_url = (
        "https://raw.githubusercontent.com/karpathy/"
        "makemore/988aa59/names.txt"
    )
    urllib.request.urlretrieve(names_url, input_path)

with open(input_path) as file:
    documents = [line.strip() for line in file if line.strip()]

random.shuffle(documents)

split_index = int(0.9 * len(documents))
training_documents = documents[:split_index]
test_documents = documents[split_index:]

print(f"number of documents: {len(documents)}")
print(f"training documents: {len(training_documents)}")
print(f"test documents: {len(test_documents)}")
print(f"first 10 documents: {documents[:10]}")

# The tokenizer assigns one integer token ID to every character.
unique_characters = sorted(set("".join(documents)))
character_to_id = {
    character: token_id
    for token_id, character in enumerate(unique_characters)
}
id_to_character = {
    token_id: character
    for character, token_id in character_to_id.items()
}

# The same special token marks both the beginning and end of a document.
BOS = len(unique_characters)
vocabulary_size = len(unique_characters) + 1


def encode(document):
    return [BOS] + [character_to_id[character] for character in document] + [BOS]


def decode(token_ids):
    return "".join(
        id_to_character[token_id]
        for token_id in token_ids
        if token_id != BOS
    )


def run_fast_training():
    import time

    import torch
    import torch.nn.functional as F

    # This model is too small to benefit from GPU kernel launches. One CPU
    # thread is considerably faster for its many tiny tensor operations.
    torch.set_num_threads(1)

    device_name = os.environ.get("DEVICE", "cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("DEVICE=mps was requested, but MPS is unavailable")
    device = torch.device(device_name)

    fast_embedding_size = 16
    fast_block_size = 16
    fast_number_of_heads = 4
    fast_head_size = fast_embedding_size // fast_number_of_heads

    class FastMicroGPT(torch.nn.Module):
        def __init__(self):
            super().__init__()

            def make_parameter(number_of_rows, number_of_columns):
                values = [
                    [
                        random.gauss(0.0, 0.08)
                        for _ in range(number_of_columns)
                    ]
                    for _ in range(number_of_rows)
                ]
                return torch.nn.Parameter(
                    torch.tensor(values, dtype=torch.float32)
                )

            self.token_embedding_table = make_parameter(
                vocabulary_size,
                fast_embedding_size,
            )
            self.position_embedding_table = make_parameter(
                fast_block_size,
                fast_embedding_size,
            )

            self.query_weights = make_parameter(
                fast_embedding_size,
                fast_embedding_size,
            )
            self.key_weights = make_parameter(
                fast_embedding_size,
                fast_embedding_size,
            )
            self.value_weights = make_parameter(
                fast_embedding_size,
                fast_embedding_size,
            )
            self.attention_output_weights = make_parameter(
                fast_embedding_size,
                fast_embedding_size,
            )

            self.mlp_expand_weights = make_parameter(
                4 * fast_embedding_size,
                fast_embedding_size,
            )
            self.mlp_contract_weights = make_parameter(
                fast_embedding_size,
                4 * fast_embedding_size,
            )
            self.language_model_head_weights = make_parameter(
                vocabulary_size,
                fast_embedding_size,
            )

            causal_mask = torch.triu(
                torch.ones(
                    fast_block_size,
                    fast_block_size,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            self.register_buffer(
                "causal_mask",
                causal_mask,
                persistent=False,
            )

        def forward(self, token_ids):
            sequence_length = token_ids.shape[0]

            inputs = (
                self.token_embedding_table[token_ids]
                + self.position_embedding_table[:sequence_length]
            )

            normalized_inputs = F.rms_norm(
                inputs,
                (fast_embedding_size,),
                eps=1e-5,
            )
            query = F.linear(normalized_inputs, self.query_weights)
            key = F.linear(normalized_inputs, self.key_weights)
            value = F.linear(normalized_inputs, self.value_weights)

            query = query.view(
                sequence_length,
                fast_number_of_heads,
                fast_head_size,
            ).transpose(0, 1)
            key = key.view(
                sequence_length,
                fast_number_of_heads,
                fast_head_size,
            ).transpose(0, 1)
            value = value.view(
                sequence_length,
                fast_number_of_heads,
                fast_head_size,
            ).transpose(0, 1)

            attention_logits = query @ key.transpose(-2, -1)
            attention_logits = attention_logits / fast_head_size**0.5
            attention_logits = attention_logits.masked_fill(
                self.causal_mask[:sequence_length, :sequence_length],
                float("-inf"),
            )
            attention_weights = F.softmax(attention_logits, dim=-1)

            concatenated_heads = attention_weights @ value
            concatenated_heads = concatenated_heads.transpose(0, 1).reshape(
                sequence_length,
                fast_embedding_size,
            )
            attention_update = F.linear(
                concatenated_heads,
                self.attention_output_weights,
            )
            after_attention = inputs + attention_update

            normalized_after_attention = F.rms_norm(
                after_attention,
                (fast_embedding_size,),
                eps=1e-5,
            )
            hidden = F.linear(
                normalized_after_attention,
                self.mlp_expand_weights,
            )
            hidden = F.relu(hidden)
            mlp_update = F.linear(hidden, self.mlp_contract_weights)
            block_output = after_attention + mlp_update

            return F.linear(
                block_output,
                self.language_model_head_weights,
            )

    def make_training_pair(document):
        token_ids = encode(document)
        number_of_predictions = min(
            fast_block_size,
            len(token_ids) - 1,
        )
        input_ids = torch.tensor(
            token_ids[:number_of_predictions],
            dtype=torch.long,
            device=device,
        )
        target_ids = torch.tensor(
            token_ids[1 : number_of_predictions + 1],
            dtype=torch.long,
            device=device,
        )
        return input_ids, target_ids

    @torch.inference_mode()
    def evaluate_model(model, evaluation_documents):
        model.eval()
        total_loss = 0.0
        total_predictions = 0

        for document_index, document in enumerate(evaluation_documents):
            input_ids, target_ids = make_training_pair(document)
            logits = model(input_ids)
            total_loss += F.cross_entropy(
                logits,
                target_ids,
                reduction="sum",
            ).item()
            total_predictions += target_ids.numel()

            number_evaluated = document_index + 1
            if number_evaluated % 500 == 0 or number_evaluated == len(
                evaluation_documents
            ):
                print(
                    f"evaluated {number_evaluated}/"
                    f"{len(evaluation_documents)} test documents"
                )

        average_loss = total_loss / total_predictions
        perplexity = math.exp(average_loss)
        return average_loss, perplexity, total_predictions

    def make_prefix_tensor(prefix):
        unknown_characters = sorted(
            set(prefix) - set(character_to_id)
        )
        if unknown_characters:
            raise ValueError(
                f"prompt contains unknown characters: {unknown_characters}"
            )
        if len(prefix) >= fast_block_size:
            raise ValueError(
                f"prompt must be shorter than {fast_block_size} characters"
            )

        token_ids = [BOS] + [
            character_to_id[character] for character in prefix
        ]
        return torch.tensor(
            token_ids,
            dtype=torch.long,
            device=device,
        )

    @torch.inference_mode()
    def predict_next_character(model, prefix, temperature):
        model.eval()
        prefix_ids = make_prefix_tensor(prefix)
        next_token_logits = model(prefix_ids)[-1] / temperature
        probabilities = F.softmax(next_token_logits, dim=-1)

        top_probabilities, top_token_ids = torch.topk(probabilities, k=5)
        top_predictions = [
            (
                "<END>" if token_id.item() == BOS else id_to_character[
                    token_id.item()
                ],
                probability.item(),
            )
            for probability, token_id in zip(
                top_probabilities,
                top_token_ids,
            )
        ]
        return probabilities, top_predictions

    @torch.inference_mode()
    def generate_name(model, prefix, temperature):
        generated_token_ids = [BOS] + [
            character_to_id[character] for character in prefix
        ]

        for _ in range(fast_block_size - len(prefix)):
            input_ids = torch.tensor(
                generated_token_ids,
                dtype=torch.long,
                device=device,
            )
            next_token_logits = model(input_ids)[-1] / temperature
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(
                probabilities,
                num_samples=1,
            ).item()

            if next_token_id == BOS:
                break
            generated_token_ids.append(next_token_id)

        return decode(generated_token_ids)

    model = FastMicroGPT().to(device)
    number_of_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert number_of_parameters == 4192

    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "microgpt_checkpoint.pt",
    )
    run_mode = os.environ.get("MODE", "train").lower()

    if run_mode == "generate":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                "no checkpoint found; train the model before generating"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        prompt = os.environ.get("PROMPT", "").strip().lower()
        temperature = float(os.environ.get("TEMPERATURE", "0.8"))
        number_of_samples = int(os.environ.get("NUM_SAMPLES", "20"))
        sample_seed = int(os.environ.get("SAMPLE_SEED", "42"))

        if temperature <= 0.0:
            raise ValueError("TEMPERATURE must be greater than zero")
        if number_of_samples <= 0:
            raise ValueError("NUM_SAMPLES must be greater than zero")

        torch.manual_seed(sample_seed)
        _, top_predictions = predict_next_character(
            model,
            prompt,
            temperature,
        )

        displayed_prompt = prompt if prompt else "<BOS>"
        print(f"checkpoint training steps: {checkpoint['training_steps']}")
        print(f"prompt: {displayed_prompt!r}")
        print(
            "top next-character predictions:",
            [
                (character, round(probability, 4))
                for character, probability in top_predictions
            ],
        )
        print("generated names:")
        for _ in range(number_of_samples):
            print(f"  {generate_name(model, prompt, temperature)}")
        return

    if run_mode != "train":
        raise ValueError("MODE must be either 'train' or 'generate'")

    number_of_training_steps = int(
        os.environ.get("TRAINING_STEPS", "20")
    )
    evaluation_limit = int(os.environ.get("EVALUATION_LIMIT", "0"))
    if number_of_training_steps < 0:
        raise ValueError("TRAINING_STEPS cannot be negative")
    if evaluation_limit < 0:
        raise ValueError("EVALUATION_LIMIT cannot be negative")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        betas=(0.85, 0.99),
        eps=1e-8,
        foreach=False,
    )

    print(f"training engine: PyTorch tensors on {device}")
    print(f"number of trainable parameters: {number_of_parameters}")
    print(f"training steps: {number_of_training_steps}")

    model.train()
    training_started = time.perf_counter()
    report_interval = max(1, number_of_training_steps // 20)
    recent_loss = 0.0
    steps_since_report = 0

    for training_iteration in range(number_of_training_steps):
        step_number = training_iteration + 1
        document_index = training_iteration % len(training_documents)
        document = training_documents[document_index]
        input_ids, target_ids = make_training_pair(document)

        logits = model(input_ids)
        loss = F.cross_entropy(logits, target_ids)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        progress = training_iteration / number_of_training_steps
        learning_rate = 0.01 * (1.0 - progress)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.step()

        recent_loss += loss.item()
        steps_since_report += 1

        if step_number % report_interval == 0 or (
            step_number == number_of_training_steps
        ):
            elapsed = time.perf_counter() - training_started
            average_recent_loss = recent_loss / steps_since_report
            print(
                f"step {step_number:>6}/{number_of_training_steps}",
                f"loss={average_recent_loss:.4f}",
                f"lr={learning_rate:.8f}",
                f"speed={step_number / elapsed:.0f} names/s",
            )
            recent_loss = 0.0
            steps_since_report = 0

    training_seconds = time.perf_counter() - training_started

    documents_to_evaluate = (
        test_documents
        if evaluation_limit == 0
        else test_documents[:evaluation_limit]
    )
    test_loss, test_perplexity, test_predictions = evaluate_model(
        model,
        documents_to_evaluate,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_steps": number_of_training_steps,
            "character_to_id": character_to_id,
            "id_to_character": id_to_character,
            "config": {
                "vocabulary_size": vocabulary_size,
                "embedding_size": fast_embedding_size,
                "block_size": fast_block_size,
                "number_of_attention_heads": fast_number_of_heads,
            },
        },
        checkpoint_path,
    )

    print(f"training time: {training_seconds:.2f} seconds")
    print(f"test documents evaluated: {len(documents_to_evaluate)}")
    print(f"next-token predictions evaluated: {test_predictions}")
    print(f"test loss: {test_loss:.4f}")
    print(f"test perplexity: {test_perplexity:.2f}")
    print(f"checkpoint saved to: {checkpoint_path}")


engine = os.environ.get("ENGINE", "torch").lower()
if engine == "torch":
    run_fast_training()
    raise SystemExit
if engine != "scalar":
    raise ValueError("ENGINE must be either 'torch' or 'scalar'")


example_document = training_documents[0]
example_tokens = encode(example_document)

assert decode(example_tokens) == example_document

print(f"vocabulary: {unique_characters}")
print(f"vocabulary size: {vocabulary_size}")
print(f"encoded {example_document!r}: {example_tokens}")
print(f"decoded tokens: {decode(example_tokens)!r}")


class Value:
    def __init__(self, data, children=(), local_gradients=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_gradients = local_gradients

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data + other.data,
            children=(self, other),
            local_gradients=(1.0, 1.0),
        )

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data * other.data,
            children=(self, other),
            local_gradients=(other.data, self.data),
        )

    def __pow__(self, exponent):
        return Value(
            self.data**exponent,
            children=(self,),
            local_gradients=(exponent * self.data ** (exponent - 1),),
        )

    def log(self):
        return Value(
            math.log(self.data),
            children=(self,),
            local_gradients=(1.0 / self.data,),
        )

    def exp(self):
        result = math.exp(self.data)
        return Value(
            result,
            children=(self,),
            local_gradients=(result,),
        )

    def relu(self):
        return Value(
            max(0.0, self.data),
            children=(self,),
            local_gradients=(float(self.data > 0.0),),
        )

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def backward(self):
        topological_order = []
        visited = set()

        def build_topological_order(value):
            if value in visited:
                return

            visited.add(value)
            for child in value._children:
                build_topological_order(child)
            topological_order.append(value)

        build_topological_order(self)

        self.grad = 1.0
        for value in reversed(topological_order):
            for child, local_gradient in zip(
                value._children,
                value._local_gradients,
            ):
                child.grad += local_gradient * value.grad


a = Value(2.0)
b = Value(-3.0)
m = a * b
c = m + a

assert c.data == -4.0
c.backward()

assert c.grad == 1.0
assert m.grad == 1.0
assert a.grad == -2.0
assert b.grad == 2.0

print(f"a: {a}")
print(f"b: {b}")
print(f"m: {m}")
print(f"c: {c}")

x = Value(2.0)
y = (3.0 / x) - x
y.backward()

assert math.isclose(y.data, -0.5)
assert math.isclose(x.grad, -1.75)

print(f"extended-operations example: y={y}, x={x}")


def linear(inputs, weights):
    return [
        sum(
            weight * input_value
            for weight, input_value in zip(output_weights, inputs)
        )
        for output_weights in weights
    ]


linear_inputs = [Value(2.0), Value(3.0)]
linear_weights = [
    [Value(1.0), Value(-1.0)],
    [Value(0.5), Value(2.0)],
]
linear_outputs = linear(linear_inputs, linear_weights)

assert [output.data for output in linear_outputs] == [-1.0, 7.0]

linear_outputs[0].backward()
assert [input_value.grad for input_value in linear_inputs] == [1.0, -1.0]

print(f"linear outputs: {linear_outputs}")


def softmax(logits):
    maximum_logit = max(logit.data for logit in logits)
    exponentials = [(logit - maximum_logit).exp() for logit in logits]
    total = sum(exponentials)
    return [exponential / total for exponential in exponentials]


example_logits = [Value(1.0), Value(2.0), Value(3.0)]
example_probabilities = softmax(example_logits)

assert math.isclose(
    sum(probability.data for probability in example_probabilities),
    1.0,
)

print(
    "softmax probabilities:",
    [round(probability.data, 4) for probability in example_probabilities],
)


def rms_norm(inputs):
    mean_square = sum(input_value * input_value for input_value in inputs)
    mean_square = mean_square / len(inputs)
    scale = (mean_square + 1e-5) ** -0.5
    return [input_value * scale for input_value in inputs]


rms_inputs = [Value(3.0), Value(4.0)]
rms_outputs = rms_norm(rms_inputs)
rms_output_magnitude = math.sqrt(
    sum(output.data**2 for output in rms_outputs) / len(rms_outputs)
)

assert math.isclose(rms_output_magnitude, 1.0, rel_tol=1e-6)

print(
    "RMS-normalized values:",
    [round(output.data, 4) for output in rms_outputs],
)
print(f"RMS magnitude: {rms_output_magnitude:.6f}")


embedding_size = 16
block_size = 16


def make_matrix(number_of_rows, number_of_columns, standard_deviation=0.08):
    return [
        [
            Value(random.gauss(0.0, standard_deviation))
            for _ in range(number_of_columns)
        ]
        for _ in range(number_of_rows)
    ]


token_embedding_table = make_matrix(vocabulary_size, embedding_size)
position_embedding_table = make_matrix(block_size, embedding_size)


def embed(token_id, position_id):
    token_embedding = token_embedding_table[token_id]
    position_embedding = position_embedding_table[position_id]
    return [
        token_value + position_value
        for token_value, position_value in zip(
            token_embedding,
            position_embedding,
        )
    ]


example_embedding = embed(BOS, 0)

assert len(example_embedding) == embedding_size
assert all(isinstance(value, Value) for value in example_embedding)

print(
    "BOS embedding at position 0:",
    [round(value.data, 4) for value in example_embedding[:4]],
    "...",
)


number_of_attention_heads = 4
head_size = embedding_size // number_of_attention_heads

assert embedding_size % number_of_attention_heads == 0

query_weights = make_matrix(embedding_size, embedding_size)
key_weights = make_matrix(embedding_size, embedding_size)
value_weights = make_matrix(embedding_size, embedding_size)


def project_query_key_value(inputs):
    normalized_inputs = rms_norm(inputs)
    query = linear(normalized_inputs, query_weights)
    key = linear(normalized_inputs, key_weights)
    value = linear(normalized_inputs, value_weights)
    return query, key, value


example_query, example_key, example_value = project_query_key_value(
    example_embedding
)

assert len(example_query) == embedding_size
assert len(example_key) == embedding_size
assert len(example_value) == embedding_size

print(f"attention heads: {number_of_attention_heads}")
print(f"dimensions per head: {head_size}")
print(
    "first query head:",
    [round(value.data, 4) for value in example_query[:head_size]],
)


def attention_head(query, keys, values, head_index):
    assert len(keys) == len(values)

    head_start = head_index * head_size
    head_end = head_start + head_size

    query_head = query[head_start:head_end]
    key_heads = [key[head_start:head_end] for key in keys]
    value_heads = [value[head_start:head_end] for value in values]

    attention_logits = [
        sum(
            query_component * key_component
            for query_component, key_component in zip(query_head, key_head)
        )
        / head_size**0.5
        for key_head in key_heads
    ]
    attention_weights = softmax(attention_logits)

    head_output = [
        sum(
            attention_weight * value_head[dimension]
            for attention_weight, value_head in zip(
                attention_weights,
                value_heads,
            )
        )
        for dimension in range(head_size)
    ]

    return head_output, attention_weights


two_token_ids = encode(example_document)[:2]
cached_keys = []
cached_values = []

for position_id, token_id in enumerate(two_token_ids):
    token_embedding = embed(token_id, position_id)
    query, key, value = project_query_key_value(token_embedding)
    cached_keys.append(key)
    cached_values.append(value)

first_head_output, first_head_weights = attention_head(
    query,
    cached_keys,
    cached_values,
    head_index=0,
)

assert len(first_head_output) == head_size
assert math.isclose(
    sum(weight.data for weight in first_head_weights),
    1.0,
)

print(
    "first-head attention weights:",
    [round(weight.data, 4) for weight in first_head_weights],
)


attention_output_weights = make_matrix(embedding_size, embedding_size)


def multi_head_attention(query, keys, values):
    concatenated_head_outputs = []
    weights_by_head = []

    for head_index in range(number_of_attention_heads):
        head_output, attention_weights = attention_head(
            query,
            keys,
            values,
            head_index,
        )
        concatenated_head_outputs.extend(head_output)
        weights_by_head.append(attention_weights)

    output = linear(concatenated_head_outputs, attention_output_weights)
    return output, weights_by_head


example_attention_output, example_weights_by_head = multi_head_attention(
    query,
    cached_keys,
    cached_values,
)

assert len(example_attention_output) == embedding_size
assert len(example_weights_by_head) == number_of_attention_heads

print(
    "multi-head attention output:",
    [round(value.data, 4) for value in example_attention_output[:4]],
    "...",
)


def add_residual(inputs, update):
    assert len(inputs) == len(update)
    return [
        input_value + update_value
        for input_value, update_value in zip(inputs, update)
    ]


current_token_embedding = token_embedding
attention_residual_output = add_residual(
    current_token_embedding,
    example_attention_output,
)

assert len(attention_residual_output) == embedding_size

print(
    "after attention residual:",
    [round(value.data, 4) for value in attention_residual_output[:4]],
    "...",
)


mlp_hidden_size = 4 * embedding_size
mlp_expand_weights = make_matrix(mlp_hidden_size, embedding_size)
mlp_contract_weights = make_matrix(embedding_size, mlp_hidden_size)


def mlp(inputs):
    normalized_inputs = rms_norm(inputs)
    hidden = linear(normalized_inputs, mlp_expand_weights)
    activated = [hidden_value.relu() for hidden_value in hidden]
    return linear(activated, mlp_contract_weights)


mlp_update = mlp(attention_residual_output)
transformer_block_output = add_residual(
    attention_residual_output,
    mlp_update,
)

assert len(mlp_update) == embedding_size
assert len(transformer_block_output) == embedding_size

print(f"MLP dimensions: {embedding_size} -> {mlp_hidden_size} -> {embedding_size}")
print(
    "transformer block output:",
    [round(value.data, 4) for value in transformer_block_output[:4]],
    "...",
)


language_model_head_weights = make_matrix(vocabulary_size, embedding_size)


def predict_next_token(inputs):
    logits = linear(inputs, language_model_head_weights)
    probabilities = softmax(logits)
    return logits, probabilities


example_next_token_logits, example_next_token_probabilities = predict_next_token(
    transformer_block_output
)

assert len(example_next_token_logits) == vocabulary_size
assert len(example_next_token_probabilities) == vocabulary_size
assert math.isclose(
    sum(probability.data for probability in example_next_token_probabilities),
    1.0,
)


def token_label(token_id):
    return "<BOS>" if token_id == BOS else id_to_character[token_id]


top_predictions = sorted(
    enumerate(example_next_token_probabilities),
    key=lambda item: item[1].data,
    reverse=True,
)[:5]

example_target_id = encode(example_document)[2]

print(
    "top next-token predictions:",
    [
        (token_label(token_id), round(probability.data, 4))
        for token_id, probability in top_predictions
    ],
)
print(
    "correct next token:",
    token_label(example_target_id),
    "probability:",
    round(example_next_token_probabilities[example_target_id].data, 4),
)


def negative_log_likelihood(probabilities, target_id):
    target_probability = probabilities[target_id]
    return -target_probability.log()


example_loss = negative_log_likelihood(
    example_next_token_probabilities,
    example_target_id,
)

assert example_loss.data > 0.0

example_loss.backward()

assert example_loss.grad == 1.0
assert example_next_token_logits[example_target_id].grad < 0.0

print(f"next-token loss: {example_loss.data:.4f}")
print(
    "gradient of correct-token logit:",
    round(example_next_token_logits[example_target_id].grad, 4),
)


def gpt_step(token_id, position_id, key_cache, value_cache):
    inputs = embed(token_id, position_id)

    query, key, value = project_query_key_value(inputs)
    key_cache.append(key)
    value_cache.append(value)

    attention_update, _ = multi_head_attention(
        query,
        key_cache,
        value_cache,
    )
    after_attention = add_residual(inputs, attention_update)

    mlp_update = mlp(after_attention)
    block_output = add_residual(after_attention, mlp_update)

    return predict_next_token(block_output)


step_key_cache = []
step_value_cache = []

for position_id, token_id in enumerate(two_token_ids):
    step_logits, step_probabilities = gpt_step(
        token_id,
        position_id,
        step_key_cache,
        step_value_cache,
    )

assert len(step_key_cache) == len(two_token_ids)
assert len(step_value_cache) == len(two_token_ids)
assert len(step_logits) == vocabulary_size
assert math.isclose(
    sum(probability.data for probability in step_probabilities),
    1.0,
)
assert math.isclose(
    step_probabilities[example_target_id].data,
    example_next_token_probabilities[example_target_id].data,
)

print(
    "gpt_step correct-token probability:",
    round(step_probabilities[example_target_id].data, 4),
)


def calculate_document_loss(document):
    token_ids = encode(document)
    number_of_predictions = min(block_size, len(token_ids) - 1)

    key_cache = []
    value_cache = []
    token_losses = []

    for position_id in range(number_of_predictions):
        input_id = token_ids[position_id]
        target_id = token_ids[position_id + 1]

        _, probabilities = gpt_step(
            input_id,
            position_id,
            key_cache,
            value_cache,
        )

        token_loss = negative_log_likelihood(
            probabilities,
            target_id,
        )
        token_losses.append(token_loss)

    average_loss = sum(token_losses) / number_of_predictions
    return average_loss, token_losses


example_document_loss, example_token_losses = calculate_document_loss(
    example_document
)
example_document_token_ids = encode(example_document)

assert len(example_token_losses) == len(example_document_token_ids) - 1
assert example_document_loss.data > 0.0

print(f"full-document training pairs for {example_document!r}:")
for position_id, token_loss in enumerate(example_token_losses):
    input_id = example_document_token_ids[position_id]
    target_id = example_document_token_ids[position_id + 1]
    print(
        f"  {token_label(input_id):>5} -> {token_label(target_id):<5}",
        f"loss={token_loss.data:.4f}",
    )

print(f"average document loss: {example_document_loss.data:.4f}")


parameter_matrices = [
    token_embedding_table,
    position_embedding_table,
    query_weights,
    key_weights,
    value_weights,
    attention_output_weights,
    mlp_expand_weights,
    mlp_contract_weights,
    language_model_head_weights,
]

parameters = [
    parameter
    for matrix in parameter_matrices
    for row in matrix
    for parameter in row
]

assert len(parameters) == 4192

initial_parameter_values = [parameter.data for parameter in parameters]

for parameter in parameters:
    parameter.grad = 0.0

training_loss_before_update, _ = calculate_document_loss(example_document)
training_loss_before_update.backward()

gradient_norm = math.sqrt(
    sum(parameter.grad**2 for parameter in parameters)
)

learning_rate = 0.01
example_parameter_index = max(
    range(len(parameters)),
    key=lambda index: abs(parameters[index].grad),
)
example_parameter = parameters[example_parameter_index]
example_parameter_before_update = example_parameter.data
example_parameter_gradient = example_parameter.grad

for parameter in parameters:
    parameter.data -= learning_rate * parameter.grad

example_parameter_after_update = example_parameter.data
training_loss_after_update, _ = calculate_document_loss(example_document)

assert training_loss_after_update.data < training_loss_before_update.data

print(f"number of trainable parameters: {len(parameters)}")
print(f"gradient norm: {gradient_norm:.4f}")
print(
    f"example parameter {example_parameter_index} update:",
    f"{example_parameter_before_update:.6f}",
    "-",
    f"{learning_rate} * {example_parameter_gradient:.6f}",
    "=",
    f"{example_parameter_after_update:.6f}",
)
print(
    "one-step document loss:",
    f"{training_loss_before_update.data:.4f}",
    "->",
    f"{training_loss_after_update.data:.4f}",
)


adam_beta1 = 0.85
adam_beta2 = 0.99
adam_epsilon = 1e-8
adam_first_moments = [0.0] * len(parameters)
adam_second_moments = [0.0] * len(parameters)


def apply_adam(parameters, step_number, learning_rate):
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad

        adam_first_moments[index] = (
            adam_beta1 * adam_first_moments[index]
            + (1.0 - adam_beta1) * gradient
        )
        adam_second_moments[index] = (
            adam_beta2 * adam_second_moments[index]
            + (1.0 - adam_beta2) * gradient**2
        )

        corrected_first_moment = adam_first_moments[index] / (
            1.0 - adam_beta1**step_number
        )
        corrected_second_moment = adam_second_moments[index] / (
            1.0 - adam_beta2**step_number
        )

        parameter.data -= learning_rate * corrected_first_moment / (
            corrected_second_moment**0.5 + adam_epsilon
        )


for parameter in parameters:
    parameter.grad = 0.0

adam_loss_before_update, _ = calculate_document_loss(example_document)
adam_loss_before_update.backward()

adam_example_parameter_index = max(
    range(len(parameters)),
    key=lambda index: abs(parameters[index].grad),
)
adam_example_parameter = parameters[adam_example_parameter_index]
adam_example_parameter_before = adam_example_parameter.data

apply_adam(
    parameters,
    step_number=1,
    learning_rate=0.01,
)

adam_example_parameter_after = adam_example_parameter.data
adam_loss_after_update, _ = calculate_document_loss(example_document)

assert adam_loss_after_update.data < adam_loss_before_update.data

print(
    f"Adam example parameter {adam_example_parameter_index}:",
    f"{adam_example_parameter_before:.6f}",
    "->",
    f"{adam_example_parameter_after:.6f}",
)
print(
    "one Adam step document loss:",
    f"{adam_loss_before_update.data:.4f}",
    "->",
    f"{adam_loss_after_update.data:.4f}",
)


# The SGD and Adam updates above were demonstrations. Restore the original
# model so the real training run starts from a clean initialization.
for parameter, initial_value in zip(parameters, initial_parameter_values):
    parameter.data = initial_value
    parameter.grad = 0.0

for parameter_index in range(len(parameters)):
    adam_first_moments[parameter_index] = 0.0
    adam_second_moments[parameter_index] = 0.0


number_of_training_steps = int(os.environ.get("TRAINING_STEPS", "20"))
starting_learning_rate = 0.01

for training_iteration in range(number_of_training_steps):
    step_number = training_iteration + 1
    document_index = training_iteration % len(training_documents)
    document = training_documents[document_index]

    for parameter in parameters:
        parameter.grad = 0.0

    loss, _ = calculate_document_loss(document)
    loss.backward()

    progress = training_iteration / number_of_training_steps
    learning_rate = starting_learning_rate * (1.0 - progress)
    apply_adam(parameters, step_number, learning_rate)

    print(
        f"training step {step_number:>2}",
        f"document={document!r:<14}",
        f"loss={loss.data:.4f}",
        f"learning rate={learning_rate:.5f}",
    )


def evaluate_model(evaluation_documents):
    total_loss = 0.0
    total_predictions = 0

    for document_index, document in enumerate(evaluation_documents):
        _, token_losses = calculate_document_loss(document)

        total_loss += sum(token_loss.data for token_loss in token_losses)
        total_predictions += len(token_losses)

        number_evaluated = document_index + 1
        if number_evaluated % 500 == 0 or number_evaluated == len(
            evaluation_documents
        ):
            print(
                f"evaluated {number_evaluated}/{len(evaluation_documents)}",
                "test documents",
            )

    average_loss = total_loss / total_predictions
    perplexity = math.exp(average_loss)
    return average_loss, perplexity, total_predictions


evaluation_limit = int(os.environ.get("EVALUATION_LIMIT", "0"))
documents_to_evaluate = (
    test_documents
    if evaluation_limit == 0
    else test_documents[:evaluation_limit]
)

test_loss, test_perplexity, test_predictions = evaluate_model(
    documents_to_evaluate
)

print(f"test documents evaluated: {len(documents_to_evaluate)}")
print(f"next-token predictions evaluated: {test_predictions}")
print(f"test loss: {test_loss:.4f}")
print(f"test perplexity: {test_perplexity:.2f}")
