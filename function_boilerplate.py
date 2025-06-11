from typing import List, Dict, Optional, Tuple


def process_records(
    records: List[Dict[str, str]], threshold: float
) -> Tuple[int, Optional[str]]:
    """
    Processes a list of records and returns a count of valid entries and an optional error message.

    Each record is expected to be a dictionary with string keys and string values.
    Records are considered valid if they meet certain internal criteria (e.g., numeric field exceeds a threshold).

    Args:
        records (List[Dict[str, str]]): A list of dictionaries representing individual data records.
        threshold (float): A threshold value used to filter records.

    Returns:
        Tuple[int, Optional[str]]:
            - The number of valid records that passed the threshold check.
            - An optional error message if any record was invalid or could not be processed.

    Raises:
        ValueError: If the threshold is negative.
        KeyError: If a required key is missing in any record.

    Example:
        >>> records = [{'score': '85'}, {'score': '90'}]
        >>> process_records(records, threshold=80)
        (2, None)
    """
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")

    count = 0
    for record in records:
        try:
            score = float(record["score"])
            if score >= threshold:
                count += 1
        except KeyError:
            return 0, "Missing 'score' key in one or more records."
        except ValueError:
            return 0, "Invalid score value in record."

    return count, None
