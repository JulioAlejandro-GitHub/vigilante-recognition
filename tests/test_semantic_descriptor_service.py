from app.consumer import load_fixture_message
from app.services.semantic_descriptor_service import SemanticDescriptorService


def test_semantic_descriptor_generation_for_low_quality_fixture():
    service = SemanticDescriptorService()
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")

    descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    assert descriptor.descriptor["appearance"]["dominant_palette"]
    assert descriptor.descriptor["silhouette"]["frame_aspect_ratio"] in {"portrait", "square", "landscape"}
    assert descriptor.source_frame_ref.endswith("face_low_quality.jpg")


def test_semantic_descriptor_similarity_prefers_same_image_signature():
    service = SemanticDescriptorService()
    no_face_fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    recurrent_fixture = load_fixture_message("tests/fixtures/frame_recurrent_unresolved.json")
    identified_fixture = load_fixture_message("tests/fixtures/frame_ingested_identified.json")

    no_face_descriptor = service.generate(frame_ref=no_face_fixture.frame_ref)
    recurrent_descriptor = service.generate(frame_ref=recurrent_fixture.frame_ref)
    identified_descriptor = service.generate(frame_ref=identified_fixture.frame_ref)

    same_signature_similarity = service.compare(no_face_descriptor, recurrent_descriptor)
    different_signature_similarity = service.compare(no_face_descriptor, identified_descriptor)

    assert same_signature_similarity >= 0.95
    assert same_signature_similarity > different_signature_similarity
