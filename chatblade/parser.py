import sys
import os
import argparse

from . import utils


def get_piped_input():
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


def get_api_key(options):
    if options["api_key"]:
        return options["api_key"]
    else:
        return os.environ.get("GEMINI_API_KEY", None)


# https://platform.openai.com/docs/models/continuous-model-upgrades
model_mappings = {
    "flash-2.0": "gemini-2.0-flash-exp",
}

DEFAULT_MODEL = "gemini-2.0-flash-exp"


def get_model(options):
    choice = options["model"]
    if not choice:
        return os.environ.get("GEMINI_API_MODEL", DEFAULT_MODEL)
    return model_mappings.get(choice, choice)


def get_theme(options):
    if options["theme"]:
        return options["theme"]
    return os.environ.get("CHATBLADE_THEME", None)


def extract_query(query):
    """The query comes from both the query and any piped input
    The user can provide either only a query, a piped input, or both
    in which case the piped input gets placed above the query
    """
    query = " ".join(query) if query else None
    piped_input = get_piped_input()
    if query and piped_input:
        return piped_input + "\n----------------\n" + query
    elif query:
        return query
    elif piped_input:
        return piped_input
    else:
        return None


def extract_options(options):
    options = vars(options)  # to map
    options["api_key"] = get_api_key(options)
    options["theme"] = get_theme(options)
    options["model"] = get_model(options)
    del options["query"]
    del options["model"]
    return utils.DotDict(options)


def valid_session(sess):
    if all(char not in sess for char in ["/", "\\", "\n"]):
        return sess
    else:
        raise argparse.ArgumentTypeError(f"invalid session name {sess}")


class RenameAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.session_op = "rename"
        try:
            namespace.rename_to = valid_session(values[0])
        except argparse.ArgumentTypeError as e:
            raise argparse.ArgumentError(self, f"target: {e}")


model_mappings_str = ", ".join(["%s (%s)" % e for e in model_mappings.items()])


def parse(args):
    parser = argparse.ArgumentParser(
        "Chatblade",
        description="a CLI Swiss Army Knife for Gemini",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("query", type=str, nargs="*", help="Query to send to Gemini")

    parser.add_argument(
        "--api-key",
        metavar="key",
        type=str,
        help="the Gemini API key can also be set as env variable GEMINI_API_KEY",
    )
    parser.add_argument(
        "--temperature",
        metavar="t",
        type=float,
        help="temperature",
        default=0.0,
    )
    parser.add_argument(
        "-m",
        "--model",
        help=f"""model to use either the fully qualified model name, or
        {model_mappings_str}. Can also be set via env variable GEMINI_API_MODEL
        """,
        type=str,
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="start an interactive chat session. This will implicitly continue the conversation",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--stream",
        help="Stream the incoming text to the terminal",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        help="display what *would* be sent, how many tokens, and estimated costs",
        action="store_true",
    )
    parser.add_argument(
        "--version",
        help="display the chatblade version",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--prompt-file",
        metavar="name",
        type=str,
        help=(
            "prompt name - will load the prompt with that name at ~/.config/chatblade/name or a path to a file."
            ' It uses ~/.config/chatblade/default if no prompt is specified and the "default" file exists.'
        ),
    )

    display_opts = parser.add_argument_group("result formatting options")
    display_opts.add_argument(
        "-e",
        "--extract",
        help="extract content from response if possible (either json or code block)",
        action="store_true",
    )
    display_opts.add_argument(
        "-r",
        "--raw",
        help="print session as pure text, don't pretty print or format",
        action="store_true",
    )
    display_opts.add_argument(
        "-n",
        "--no-format",
        help="do not add pretty print formatting to output",
        action="store_true",
    )
    display_opts.add_argument(
        "-o",
        "--only",
        help="Only display the response, omit query",
        action="store_true",
    )
    display_opts.add_argument(
        "--theme",
        metavar="theme",
        type=str,
        help="Set the theme for syntax highlighting see https://pygments.org/styles/, can also be set with CHATBLADE_THEME",
    )

    session_opts = parser.add_argument_group("session options")
    session_opts.add_argument(
        "-l",
        "--last",
        dest="session",
        action="store_const",
        const=utils.scratch_session,
        help=f"alias for '-S {utils.scratch_session}', the default session if none is specified",
    )
    session_opts.add_argument(
        "-S",
        "--session",
        metavar="sess",
        type=valid_session,
        help="""initiate or continue named session""",
    )
    session_opts.add_argument(
        "--session-list",
        dest="session_op",
        action="store_const",
        const="list",
        help="list sessions",
    )
    session_opts.add_argument(
        "--session-path",
        dest="session_op",
        action="store_const",
        const="path",
        help="show path to session file",
    )
    session_opts.add_argument(
        "--session-dump",
        dest="session_op",
        action="store_const",
        const="dump",
        help="dump session to stdout",
    )
    session_opts.add_argument(
        "--session-delete",
        dest="session_op",
        action="store_const",
        const="delete",
        help="delete session",
    )
    session_opts.add_argument(
        "--session-rename",
        metavar="newsess",
        action=RenameAction,
        nargs=1,
        help="rename session",
    )

    # --- debug
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)

    options = parser.parse_args(args)
    return extract_query(options.query), extract_options(options)
