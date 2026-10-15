"""Deterministic output contract checks; not a visual factuality evaluator."""
import json
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError


def nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError('Text must not be blank')
    return value


Text = Annotated[str, Field(min_length=1), AfterValidator(nonblank)]


class StrictOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')


class Event(StrictOutput):
    action: Text
    objects: list[Text]
    scene: Text
    visual_evidence: Text
    frame_references: list[Text] = Field(min_length=1, max_length=3)
    confidence: Literal['high', 'medium', 'low']


class ClipResult(StrictOutput):
    clip_id: int = Field(ge=0)
    clip_file: Text
    actual_start_time: Text
    actual_end_time: Text
    summary: Text
    visual_information_level: Literal['high', 'medium', 'low']
    setting: Text
    main_subjects: list[Text]
    events: list[Event] = Field(max_length=3)
    uncertain_parts: list[Text]
    possible_hallucination_risks: list[Text]


class OutputValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__('Model output failed validation')


def issue(path, code, message):
    return {'path': path, 'code': code, 'message': message}


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OutputValidationError([issue([], 'duplicate_json_key',
                f'Duplicate JSON object key: {key}')])
        result[key] = value
    return result


def reject_constant(value):
    raise OutputValidationError([issue([], 'invalid_json',
        'Non-finite JSON numbers are not allowed')])


def validate_output(text: str, record: dict, frame_count: int) -> dict:
    if not 1 <= frame_count <= 8:
        raise ValueError('Validation context requires 1 to 8 input frames')
    try:
        data = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise OutputValidationError([issue([], 'invalid_json',
            f'Invalid JSON at line {error.lineno}, column {error.colno}')]) from error
    try:
        parsed = ClipResult.model_validate(data).model_dump()
    except ValidationError as error:
        raise OutputValidationError([
            issue(list(item['loc']), item['type'], item['msg'])
            for item in error.errors(include_input=False, include_url=False,
                                     include_context=False)
        ]) from error

    issues = []
    for key in ['clip_id', 'clip_file', 'actual_start_time', 'actual_end_time']:
        if parsed[key] != record[key]:
            issues.append(issue([key], 'metadata_mismatch',
                                'Value must match the source clip metadata'))
    words = len(parsed['summary'].split())
    if words > 35:
        issues.append(issue(['summary'], 'word_limit',
                            f'Summary has {words} words; maximum is 35'))
    allowed = {f'Frame {index:02d}' for index in range(frame_count)}
    low_confidence = False
    for index, event in enumerate(parsed['events']):
        words = len(event['visual_evidence'].split())
        if words > 20:
            issues.append(issue(['events', index, 'visual_evidence'], 'word_limit',
                                f'Visual evidence has {words} words; maximum is 20'))
        references = event['frame_references']
        for ref_index, ref in enumerate(references):
            if ref not in allowed:
                issues.append(issue(['events', index, 'frame_references', ref_index],
                    'invalid_frame_reference',
                    'Reference must be one of: '+', '.join(sorted(allowed))))
        if len(set(references)) != len(references):
            issues.append(issue(['events', index, 'frame_references'],
                                'duplicate_frame_reference', 'Frame references must be unique'))
        low_confidence |= event['confidence'] == 'low'
    if low_confidence and not parsed['uncertain_parts']:
        issues.append(issue(['uncertain_parts'], 'missing_uncertainty',
                            'Low-confidence events require a description of uncertainty'))
    if (low_confidence or parsed['uncertain_parts']) and not parsed['possible_hallucination_risks']:
        issues.append(issue(['possible_hallucination_risks'], 'missing_risk',
                            'Explicit uncertainty requires at least one hallucination risk'))
    if issues:
        raise OutputValidationError(issues)
    return parsed
