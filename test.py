import pytest
from plate_localization import Locator
from plate_segmentation import Segment
from char_identification import CharIdentification

@pytest.mark.parametrize("image_path, expected_result", [
    ("function1/1.jpeg", ["冀B850AR"]),
    ("function1/2.jpg", ["京MKK555"])
])
def test_recognization(image_path, expected_result):
    locator = Locator(image_path)
    plates = locator.find_plate()
    segment = Segment(plates)
    plates_in_chars = segment.segment_plate()
    char_identification = CharIdentification(plates_in_chars)
    identified_plates = char_identification.identify_char()
    assert identified_plates == expected_result

if __name__ == "__main__":
    pytest.main()
