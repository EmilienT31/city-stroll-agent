"""Local LLM-as-judge for `custom_response_quality` (see eval_config.yaml)."""

from google import genai
from google.genai import types
from pydantic import BaseModel


class _Verdict(BaseModel):
    score: int  # 1-5
    explanation: str


def _latest_tool_result(instance):
    results = []
    for turn in (instance.get("agent_data") or {}).get("turns", []):
        for event in turn.get("events", []):
            for part in ((event.get("content") or {}).get("parts") or []):
                function_response = part.get("function_response") or part.get(
                    "functionResponse"
                )
                if function_response and function_response.get("name") == "generate_paths":
                    results.append(function_response.get("response") or {})
    return results[-1] if results else None


def validate_success_result(result):
    """Apply non-negotiable structural gates to a successful tool response."""

    if result.get("status") != "ok":
        return True, "The tool appropriately returned a non-success status."
    paths = result.get("paths") or []
    if len(paths) != 3:
        return False, "A successful result must contain exactly three paths."
    seen_place_ids = set()
    required = set(result.get("requiredPreferences") or [])
    for path in paths:
        stops = path.get("stops") or []
        if not 5 <= len(stops) <= 7:
            return False, "Every path must contain five to seven stops."
        if not 2000 <= path.get("walkingDistanceMeters", -1) <= 4000:
            return False, "Every path must be within the 2-4 km target."
        if not path.get("routeUrl"):
            return False, "Every path must include a Maps route URL."
        path_required = set()
        for stop in stops:
            place_id = stop.get("placeId")
            if not place_id or place_id in seen_place_ids:
                return False, "Place IDs must exist and cannot overlap across paths."
            seen_place_ids.add(place_id)
            path_required.update(stop.get("requiredPreferences") or [])
            if not stop.get("googleMapsUri"):
                return False, "Every stop must include a Google Maps URI."
        if not required.issubset(path_required):
            return False, "Every path must cover every explicit required preference."
    return True, "All deterministic City Stroll gates passed."


def evaluate(instance):
    reference = instance.get("reference")
    rubric = (
        "Grade the City Stroll agent's final response on a 1-5 scale (1 poor, "
        "5 excellent). A successful answer must use tool-grounded venues and Maps "
        "links, return exactly three distinct compact paths when the tool has enough "
        "data, respect explicit required preferences in every path, compare the "
        "alternatives clearly, and preserve safety or verification caveats. If the "
        "tool reports insufficient data, the answer must say so without fabricating "
        "results and suggest a useful constraint to relax."
    )
    if reference:
        rubric += (
            " The response should agree with the expected answer below; penalize "
            "factual disagreement with it."
        )
    prompt = (
        f"You are an expert QA evaluator for an enterprise AI assistant. {rubric}\n"
        f"User Prompt: {instance.get('prompt', '')}\n"
        f"Final Response: {instance.get('response', '')}\n"
    )
    if reference:
        prompt += f"Expected Answer (ground truth): {reference}\n"
    prompt += f"Full Agent Trace: {instance.get('agent_data', '')}\n"

    client = genai.Client()  # AI Studio (GEMINI_API_KEY) or Agent Platform (ADC)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,  # deterministic grading
            response_mime_type="application/json",
            response_schema=_Verdict,  # guaranteed schema-valid JSON
        ),
    )
    verdict = response.parsed
    if verdict is None:  # model returned nothing usable
        return {"score": 0, "explanation": response.text or ""}
    score = max(1, min(5, verdict.score))
    tool_result = _latest_tool_result(instance)
    if tool_result is None:
        return {
            "score": min(score, 2),
            "explanation": f"No generate_paths result found. {verdict.explanation}",
        }
    valid, gate_explanation = validate_success_result(tool_result)
    if not valid:
        score = min(score, 2)
    return {
        "score": score,
        "explanation": f"{gate_explanation} {verdict.explanation}",
    }
