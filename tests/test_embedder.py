from retriever.embedder import Embedder


def test_embedder_output_shape():
    embedder = Embedder()

    result = embedder.encode_single(
        "The Earth is approximately 4.54 billion years old."
    )

    assert result.shape == (384,)


def test_embed_multiple_texts():
    embedder = Embedder()

    texts = [
        "The Earth is round.",
        "The Sun is a star.",
        "Water freezes at 0 degrees Celsius.",
    ]

    result = embedder.encode(texts)

    assert result.shape == (3, 384)