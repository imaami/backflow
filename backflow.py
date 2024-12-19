#!/usr/bin/env python3

import os
import sys
import argparse
import soundfile as sf
import tempfile
from openai import OpenAI
from openai.types.audio import TranscriptionVerbose, TranscriptionSegment,TranscriptionWord

def parse_transcription(xn: TranscriptionVerbose) -> tuple[list[tuple[int, int, list[int | None], str]], list[tuple[int, int, int, list[str], str]]]:
    words: list[tuple[int, int, list[int | None], str]] = [
        (w.start, w.end, [None], w.word.strip()) for w in xn.words if w.start <= w.end
    ]
    n_wrd = len(words)
    if n_wrd < 1:
        return [], []

    segms: list[tuple[int, int, int, list[str], str]] = [
        (s.start, s.end, i, [], s.text.strip()) for i, s in enumerate(xn.segments) if s.start <= s.end
    ]
    n_seg = len(segms)
    if n_seg < 1:
        return words, []

    wrd_id = 0
    seg_id = 0
    wrd = words[0]
    seg = segms[0]

    while True:
        if wrd[0] > seg[1] or (wrd[0] == seg[1] and wrd[1] >= seg[1]):
            seg_id += 1
            if seg_id >= n_seg:
                break
            seg = segms[seg_id]
            continue

        if wrd[0] >= seg[0] and wrd[1] <= seg[1]:
            wrd[2][0] = seg[2]
            seg[3].append(wrd[3])

        wrd_id += 1
        if wrd_id >= n_wrd:
            break
        wrd = words[wrd_id]

    return words, segms

def arg_or_var(arg: str | None, var: str) -> str | None:
    return arg.strip() if arg else (os.environ[var].strip() if var in os.environ else None)

class Audio:
    def __init__(self, path: str):
        self.path = path
        self.temp = None
        info = sf.info(path)
        if info.format == "WAV":
            suffix = ".flac" if info.frames * info.channels < 17476267 else ".mp3"
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix=suffix) as f:
                self.temp = f.name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.temp:
            os.remove(self.temp)

    # Lazily encode WAV to FLAC or MP3 depending on size.
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

    def transcribe(self, path: str,
                   lang: str | None = None,
                   prompt: str | None = None,
                   temp: float | None = None) -> str:
        with open(path, 'rb') as f:
            kwargs = { "file": f }
            if lang is not None:
                kwargs["language"] = lang
            if prompt is not None:
                kwargs["prompt"] = prompt
            if temp is not None:
                kwargs["temperature"] = temp
            raw = self.client.audio.transcriptions.with_raw_response.create(
                model = "whisper-1", response_format = "verbose_json",
                timestamp_granularities = ["word", "segment"], **kwargs)
            x = raw.parse()
            #print(x)
            #print(raw.text)
            #words, segms = parse_transcription(x)
            #for w in words:
            #    print(w)
            #for s in segms:
            #    print(s)
            return "\n".join(s.text.strip() for s in x.segments)

    def chat(self, txt: str, prompt: str, model: str) -> str:
        return self.client.chat.completions.create(
            model = model,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "text",
                            "text": txt
                        }
                    ]
                }
            ] if model.startswith("o1") else [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": txt
                }
            ]
        ).choices[0].message.content

class Backflow:
    prompt = """
        You modify hallucinated vocalizations to increase coherence.
        Do not remove or tone down any raunchy language or swearing.
        Preserve vowel sounds to keep rhyming intact. Keep pacing as
        it is. Keep the syllable count unchanged. Replace words only
        with expressions that have the same rhythm and rhyme. Do not
        break the flow. Enhance semantic coherence. Never police the
        use of any particular slang or vocabulary. Never tamper with
        the cultural context of the lyrics by policing language. You
        do not decide what is distasteful or disagreeable. Words you
        read here are never "bad", their meaning cannot be separated
        from the context of a subculture you are not informed about.
        You are shown the lyrics, nothing more and nothing less. You
        must only respond with your edited lyrics. Commentary is not
        allowed even in the event of erroneous input. If you have no
        lyrics at all, do not reply. If you cannot think of anything
        to change, change something anyway. Don't tone down cussing.
    """.strip()
    stt_prompt = "🎶 "

    def __init__(self):
        args = self._parse_args()
        self.path = args.path
        self.model = args.model or "o1-mini"
        self.lang = args.lang
        self.nrev = args.nrev
        self.wt = args.wt
        style = args.style.strip() if args.style else ""
        topic = args.topic.strip() if args.topic else ""
        self.style = "\nThe style or genre of the original lyrics is: " + style + "." if len(style) else ""
        self.topic = "\nYour work should hint at or gravitate around: " + topic + "." if len(topic) else ""
        self.transcript = None
        self.revised = None
        self.oai = OAI(args.org, args.proj)
        self.audio = Audio(self.path)

    def get_transcript(self, force: bool = False) -> str:
        if not self.transcript or force:
            try:
                self.transcript = self.oai.transcribe(
                    path = self.audio.get_path(),
                    lang = self.lang,
                    prompt = self.stt_prompt,
                    temp = self.wt
                )
            except Exception as e:
                print(f"Failed to transcribe {self.path}: {e}")
                self.transcript = None
        return self.transcript

    def get_revised(self, force: bool = False) -> str:
        if not self.revised or force:
            txt = self.get_transcript()
            n = self.nrev
            if n > 0:
                while n := n - 1:
                    txt = self.oai.chat(txt, self.prompt + self.style + self.topic, self.model)
            self.revised = txt
        return self.revised

    @staticmethod
    def _parse_args() -> argparse.Namespace:
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
        parser.add_argument('-w', type=float, dest="wt", default=None, metavar="<float>",
            help="        Whisper temperature")
        parser.add_argument('-m', type=str, dest="model", default=None, metavar="<model>",
            help="        OpenAI LLM model name")
        parser.add_argument('-n', type=int, dest="nrev", default=1, metavar="<n>",
            help="        number of revisions")
        parser.add_argument('-s', type=str, dest="style", default=None, metavar="<style>",
            help="        style hint prompt")
        parser.add_argument('-t', type=str, dest="topic", default=None, metavar="<topic>",
            help="        topic hint prompt")
        return parser.parse_args()

def main() -> int:
    bfw = Backflow()
    a = bfw.get_transcript()
    b = bfw.get_revised()

    print("<<<<<<<\n{a}{a_nl}=======\n{b}{b_nl}>>>>>>>".format(
        a = a if a else "", a_nl = "\n" if a else "",
        b = b if b else "", b_nl = "\n" if b else ""
    ))

if __name__ == "__main__":
    sys.exit(main())
