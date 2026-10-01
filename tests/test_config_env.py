import tempfile
import unittest
from pathlib import Path

from config_env import load_environment, parse_env_file


class EnvConfigTests(unittest.TestCase):
    def test_parse_env_file_supports_comments_quotes_and_literal_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# local configuration\n"
                "PLAIN=value\n"
                "DOUBLE=\"two words\"\n"
                "SINGLE='literal $VALUE'\n"
                "EMPTY=\n",
                encoding="utf-8",
            )

            self.assertEqual(
                parse_env_file(path),
                {
                    "PLAIN": "value",
                    "DOUBLE": "two words",
                    "SINGLE": "literal $VALUE",
                    "EMPTY": "",
                },
            )

    def test_process_environment_takes_precedence_over_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("SERVER_URL=http://from-file\nONLY_FILE=yes\n", encoding="utf-8")

            environment = load_environment(path, {"SERVER_URL": "http://exported"})

            self.assertEqual(environment["SERVER_URL"], "http://exported")
            self.assertEqual(environment["ONLY_FILE"], "yes")


if __name__ == "__main__":
    unittest.main()