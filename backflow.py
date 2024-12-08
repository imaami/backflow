#!/usr/bin/env python3

import os
import sys
import argparse
import soundfile as sf
import tempfile
from openai import OpenAI

def arg_or_var(arg: str | None, var: str) -> str | None:
    return arg.strip() if arg else (os.environ[var].strip() if var in os.environ else None)

class Backflow:
    def __init__(self, path: str):
        self.path = path
        info = sf.info(path)
        if info.format == "WAV":
            suffix = ".flac" if info.frames * info.channels < 17476267 else ".mp3"
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix=suffix) as f:
                self.temp = f.name
        else:
            self.temp = None

    def __del__(self):
        if self.temp:
            os.remove(self.temp)

    # Convert WAV to FLAC.
    def get_path(self) -> str:
        if self.path:
            if not self.temp:
                return self.path
            try:
                with sf.SoundFile(self.path) as f:
                    sf.write(self.temp, f.read(), f.samplerate)
            except Exception as e:
                print(f"Failed to encode {self.path}: {e}")
                os.remove(self.temp)
                self.temp = None
                return self.path
            self.path = None
        return self.temp

class OAI:
    llm_model = "o1-mini"
    stt_prompt = "🎶 "
    llm_prompt = """
        You modify hallucinated vocalizations to increase coherence.
        You do not remove or tone down raunchy language or swearing.
        You preserve vowels so as to not break rhyming. You keep the
        pacing of the vocals unmodified. You keep the syllable count
        unchanged. You replace words only with expressions that have
        equal rhythm and rhyme. You never break the flow. You strive
        to enhance semantic coherence only. You never police the use
        of any particular slang or vocabulary. You do not substitute
        expressions with those that would break rhythm or rhyme. You
        never alter the cultural context of the lyrics by sanitizing
        them, no matter how distasteful or disagreeable, because you
        consider artistic expression sacred. Your source material is
        the original lyrics, nothing more or less. You submit edited
        lyrics, nothing else. You do not remove or hide swear words.
    """.strip()

    def __init__(self, org: str | None = None, proj: str | None = None):
        api_key = arg_or_var(None, "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is not set")
        self.client = OpenAI(
            api_key = api_key,
            organization = arg_or_var(org, "OPENAI_ORG_ID"),
            project = arg_or_var(proj, "OPENAI_PROJECT_ID")
        )
        self.txt = []

    def __del__(self):
        self.client.close()

    def transcribe(self, path: str) -> str:
        with open(path, 'rb') as f:
            return "\n".join(s.text.strip() for s in
                self.client.audio.transcriptions.create(
                    model = "whisper-1",
                    response_format = "verbose_json",
                    prompt = self.stt_prompt,
                    file = f
                ).segments)

    def chat(self, msgs: list[dict]) -> str:
        return self.client.chat.completions.create(
            model = self.llm_model,
            messages = msgs
        ).choices[0].message.content

    def revise(self, txt: str) -> str:
        return self.chat([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.llm_prompt
                    },
                    {
                        "type": "text",
                        "text": txt
                    }
                ]
            }
        ])

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="backflow.py",
        description="Turn hallucinated vocals into almost coherent lyrics",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''
Environment variables:
  OPENAI_API_KEY        OpenAI API key, mandatory
  OPENAI_ORG_ID         used if -o is not provided
  OPENAI_PROJECT_ID     used if -p is not provided
'''
    )
    parser.add_argument("path", type=str, metavar="<file>",
        help="        path to the input audio file")
    parser.add_argument('-h', '--help', action='help',
        help="        show this help text and exit")
    parser.add_argument('-o', type=str, dest="org", default=None, metavar="<org-id>",
        help="        OpenAI organization ID")
    parser.add_argument('-p', type=str, dest="proj", default=None, metavar="<proj-id>",
        help="        OpenAI project ID")
    parser.add_argument('-l', type=str, dest="lang", default=None, metavar="<code>",
        help="        ISO 639-1 language code")
    parser.add_argument('-n', type=int, dest="nrev", default=1, metavar="<n>",
        help="        number of revisions")
    parser.add_argument('-s', type=str, dest="style", default=None, metavar="<style>",
        help="        style hint prompt")
    parser.add_argument('-t', type=str, dest="topic", default=None, metavar="<topic>",
        help="        topic hint prompt")

    arg = parser.parse_args()
    oai = OAI(arg.org, arg.proj)
    bfw = Backflow(arg.path)

    a = oai.transcribe(bfw.get_path())
    n = arg.nrev

    if n > 0:
        b = a
        while n := n - 1:
            b = oai.revise(b)
    else:
        b = None

    print("<<<<<<<\n{a}\n=======\n{b}\n>>>>>>>".format(
        a = a if a else "",
        b = b if b else ""
    ))

if __name__ == "__main__":
    sys.exit(main())
