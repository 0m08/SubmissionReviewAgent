import unittest
from PIL import Image
from agents.submission_reviewer.image_identifier import (
    generate_image_description,
    score_description_match,
)
from agents.submission_reviewer.hvac_eval_processor import resolve_column_indices


class TestHVACEvalPipeline(unittest.TestCase):

    def test_resolve_column_indices(self):
        headers = [
            "Image Link",
            "Image Description",
            "Prompt",
            "Image description\nModel output\n(gpt 5.6 luna max)",
            "Scoring agent\nMatch (Yes / No)"
        ]
        col_map = resolve_column_indices(headers)
        self.assertEqual(col_map.get("image_link"), 1)
        self.assertEqual(col_map.get("human_desc"), 2)
        self.assertEqual(col_map.get("prompt"), 3)
        self.assertEqual(col_map.get("model_output"), 4)
        self.assertEqual(col_map.get("match_score"), 5)

    def test_agent1_no_images_fallback(self):
        res = generate_image_description(prompt="Identify HVAC component", images=[])
        self.assertIn("No images provided", res)

    def test_agent2_scoring_logic_matches(self):
        ai_desc = "Yellow manifold gauge set connected to R-410A refrigerant cylinder showing 120 PSI low pressure."
        human_desc = "Manifold gauge set displaying pressure reading on R-410A refrigerant line."
        
        match_score, rationale = score_description_match(ai_desc, human_desc, model_choice="gemini-3.7-flash")
        self.assertIn(match_score, ["Yes", "No"])
        self.assertIsNotNone(rationale)

    def test_agent2_scoring_logic_mismatch(self):
        ai_desc = "Multimeter measuring DC voltage on a battery."
        human_desc = "Digital manifold gauge attached to low pressure service port."
        
        match_score, rationale = score_description_match(ai_desc, human_desc, model_choice="gemini-3.7-flash")
        self.assertIn(match_score, ["Yes", "No"])

    def test_agent2_empty_input_fallback(self):
        match_score, rationale = score_description_match("", "Some human description")
        self.assertEqual(match_score, "No")
        self.assertIn("missing", rationale.lower())


if __name__ == "__main__":
    unittest.main()
