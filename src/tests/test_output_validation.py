import copy
import json
import unittest

from src.serving.output_validation import OutputValidationError, validate_output


def record():
    return {'clip_id': 1, 'clip_file': 'clip_1.mp4',
            'actual_start_time': '00:00:20.000', 'actual_end_time': '00:00:40.000',
            'selected_frames': [{'local_timestamp_sec': i} for i in range(8)]}


def valid_output():
    return {**{k: record()[k] for k in ['clip_id', 'clip_file', 'actual_start_time', 'actual_end_time']},
            'summary': 'A figure stands beside a door.', 'visual_information_level': 'high',
            'setting': 'A room.', 'main_subjects': ['figure'],
            'events': [{'action': 'standing', 'objects': ['figure'], 'scene': 'room',
                        'visual_evidence': 'A figure stands beside a door.',
                        'frame_references': ['Frame 00', 'Frame 07'], 'confidence': 'high'}],
            'uncertain_parts': [], 'possible_hallucination_risks': []}


class ValidationTests(unittest.TestCase):
    def validate(self, output, frames=8):
        return validate_output(json.dumps(output), record(), frames)

    def reject(self, output, code, frames=8):
        with self.assertRaises(OutputValidationError) as caught:
            self.validate(output, frames)
        self.assertIn(code, [i['code'] for i in caught.exception.issues])
        return caught.exception.issues

    def test_valid_output_is_not_rewritten(self):
        value = valid_output()
        self.assertEqual(self.validate(value), value)

    def test_word_boundaries(self):
        value = valid_output()
        value['summary'] = ' '.join(['word']*35)
        value['events'][0]['visual_evidence'] = ' '.join(['word']*20)
        self.validate(value)
        value['summary'] += ' extra'
        self.reject(value, 'word_limit')
        value['summary'] = 'Short summary.'
        value['events'][0]['visual_evidence'] += ' extra'
        issues = self.reject(value, 'word_limit')
        self.assertEqual(issues[0]['path'], ['events', 0, 'visual_evidence'])

    def test_whitespace_word_count_and_hyphenated_words(self):
        value = valid_output()
        value['summary'] = '\t\n'.join(['well-lit']*35)
        self.validate(value)
        value['summary'] += '\t extra'
        self.reject(value, 'word_limit')

    def test_frame_labels_are_zero_based_and_input_specific(self):
        value = valid_output()
        value['events'][0]['frame_references'] = ['Frame 00']
        self.validate(value, frames=1)
        value['events'][0]['frame_references'] = ['Frame 01']
        self.reject(value, 'invalid_frame_reference', frames=1)
        for label in ['Frame 08', 'Frame 99', 'Frame 1', 'frame 01', 'Frame 01 ', 'Frame -1']:
            with self.subTest(label=label):
                value['events'][0]['frame_references'] = [label]
                self.reject(value, 'invalid_frame_reference')

    def test_reference_count_and_duplicates(self):
        value = valid_output()
        value['events'][0]['frame_references'] = []
        self.reject(value, 'too_short')
        value['events'][0]['frame_references'] = ['Frame 00']*2
        self.reject(value, 'duplicate_frame_reference')
        value['events'][0]['frame_references'] = [f'Frame {i:02d}' for i in range(4)]
        self.reject(value, 'too_long')

    def test_event_limit(self):
        value = valid_output()
        value['events'] *= 3
        self.validate(value)
        value['events'].append(copy.deepcopy(value['events'][0]))
        self.reject(value, 'too_long')

    def test_low_confidence_requires_uncertainty_and_risk(self):
        value = valid_output()
        value['events'][0]['confidence'] = 'low'
        issues = self.reject(value, 'missing_uncertainty')
        self.assertIn('missing_risk', [i['code'] for i in issues])
        value['uncertain_parts'] = ['The figure is partly obscured.']
        self.reject(value, 'missing_risk')
        value['possible_hallucination_risks'] = ['The subject may be misidentified.']
        self.validate(value)

    def test_explicit_uncertainty_requires_risk(self):
        value = valid_output()
        value['uncertain_parts'] = ['The setting is unclear.']
        self.reject(value, 'missing_risk')

    def test_low_information_without_uncertainty_is_allowed(self):
        value = valid_output()
        value.update(events=[], main_subjects=[], visual_information_level='low')
        self.validate(value)

    def test_empty_strings_cannot_satisfy_required_fields(self):
        for key in ['summary', 'setting']:
            value = valid_output()
            value[key] = ' \n\t'
            self.reject(value, 'value_error')
        value = valid_output()
        value['possible_hallucination_risks'] = [' ']
        self.reject(value, 'value_error')

    def test_strict_types_and_enums(self):
        for bad_id in ['1', True, 1.0]:
            value = valid_output()
            value['clip_id'] = bad_id
            self.reject(value, 'int_type')
        value = valid_output()
        value['events'][0]['confidence'] = 'certain'
        self.reject(value, 'literal_error')
        value = valid_output()
        value['main_subjects'] = [7]
        self.reject(value, 'string_type')

    def test_unknown_and_missing_fields(self):
        value = valid_output()
        value['extra'] = 'unexpected'
        self.reject(value, 'extra_forbidden')
        value = valid_output()
        value['events'][0]['extra'] = 'unexpected'
        self.reject(value, 'extra_forbidden')
        value = valid_output()
        del value['uncertain_parts']
        self.reject(value, 'missing')

    def test_metadata_must_match(self):
        value = valid_output()
        value.update(clip_id=2, clip_file='other.mp4', actual_start_time='wrong', actual_end_time='wrong')
        issues = self.reject(value, 'metadata_mismatch')
        self.assertEqual(len(issues), 4)

    def test_multiple_rule_violations_are_reported_together(self):
        value = valid_output()
        value['summary'] = ' '.join(['word']*36)
        value['events'][0]['frame_references'] = ['Frame 99']
        issues = self.reject(value, 'word_limit')
        self.assertIn('invalid_frame_reference', [i['code'] for i in issues])

    def test_malformed_and_duplicate_json(self):
        for text in ['{', '```json\n{}\n```', '{"clip_id": NaN}']:
            with self.subTest(text=text), self.assertRaises(OutputValidationError) as caught:
                validate_output(text, record(), 8)
            self.assertEqual(caught.exception.issues[0]['code'], 'invalid_json')
        text = json.dumps(valid_output())[:-1]+', "summary": "again"}'
        with self.assertRaises(OutputValidationError) as caught:
            validate_output(text, record(), 8)
        self.assertEqual(caught.exception.issues[0]['code'], 'duplicate_json_key')
        with self.assertRaises(OutputValidationError):
            validate_output('[1,2]', record(), 8)


if __name__ == '__main__':
    unittest.main()
