import pytest
from plate_localization import Locator
from plate_segmentation import Segment
from char_identification import CharIdentification


@pytest.mark.parametrize("image_path, expected_result", [
    ("function1/1.jpeg", ["冀B850AR"]),
    ("function1/2.jpg", ["京MKK555"]),\
    ("function1/3.jpeg", ["京N888R8"]),\
    ("function1/4.jpg", ["闽JB8888"]),
    ("function1/5.jpeg", ["陕AK58W7"]),
    # ("function1/6.jpg", ["粤BZ8301"])
])
def test_function1(image_path, expected_result):
    locator = Locator(image_path)
    plates = locator.find_plate()
    segment = Segment(plates)
    plates_in_chars = segment.segment_plate()
    char_identification = CharIdentification(plates_in_chars)
    identified_plates = char_identification.identify_char()
    assert identified_plates == expected_result

@pytest.mark.parametrize("image_path, expected_result", [
    ("function2/1.jpg", ["粤BZ8301"]),
    ("function2/2.jpg", ["粤B002EX"]),\
    ("function2/3.jpeg", ["京N888R8"]),
    ("function2/4.jpeg", ["京A99999"]),
    ("function2/5.jpg", ["闽JB8888"]),
])
def test_function2(image_path, expected_result):
    locator = Locator(image_path)
    plates = locator.find_plate()
    segment = Segment(plates)
    plates_in_chars = segment.segment_plate()
    char_identification = CharIdentification(plates_in_chars)
    identified_plates = char_identification.identify_char()
    assert identified_plates == expected_result

@pytest.mark.parametrize("image_path, expected_result", [
    ("function3/1.jpg", ["BANKER"]),
    ("function3/2.jpg", ["ML5888"]),\
    ("function3/3.jpeg", ["WUDANG"]),
    ("function3/4.jpg", ["HK5888"]),
    ("function3/5.jpg", ["HH3133"]),
])
def test_function3(image_path, expected_result):
    locator = Locator(image_path)
    plates = locator.find_plate()
    segment = Segment(plates)
    plates_in_chars = segment.segment_plate()
    char_identification = CharIdentification(plates_in_chars)
    identified_plates = char_identification.identify_char()
    assert identified_plates == expected_result

if __name__ == "__main__":
    pytest.main()
