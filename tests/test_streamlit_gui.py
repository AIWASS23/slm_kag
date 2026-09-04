from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "extra gui não instalado")
class StreamlitGUITest(unittest.TestCase):
    def test_index_before_chat_flow(self) -> None:
        from streamlit.testing.v1 import AppTest

        with TemporaryDirectory() as temporary:
            database = str(Path(temporary) / "gui.db")
            environment = {
                "SLM_KAG_DB_PATH": database,
                "SLM_KAG_DOCUMENT_LOADER": "docling",
                "SLM_KAG_EXTRACTOR": "rules",
                "SLM_KAG_GENERATOR": "deterministic",
            }
            with patch.dict("os.environ", environment):
                app = AppTest.from_file(
                    str(Path(__file__).parents[1] / "main.py"), default_timeout=60
                ).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.chat_input), 0)

                app.file_uploader[0].upload(
                    "equipe.md",
                    (
                        b"# Equipe\n\nAna trabalha em Acme.\n\n"
                        b"Acme est\xc3\xa1 localizado em Fortaleza."
                    ),
                    "text/markdown",
                ).run()
                index_button = next(
                    button
                    for button in app.button
                    if button.label == "Indexar 1 documento(s)"
                )
                index_button.click().run(timeout=60)

                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, "1")
                self.assertEqual(len(app.chat_input), 1)

                app.chat_input[0].set_value("Onde Ana trabalha?").run(timeout=30)

                self.assertFalse(app.exception)
                self.assertGreaterEqual(len(app.chat_message), 2)
                self.assertTrue(
                    any(item.label.startswith("Evidências") for item in app.expander)
                )


if __name__ == "__main__":
    unittest.main()
