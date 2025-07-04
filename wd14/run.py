from typing import Generator, Iterable
from wd14.tagger.interrogator import Interrogator
from PIL import Image
from pathlib import Path

from wd14.tagger.interrogators import interrogators

tagger_args = {}
tagger_args["threshold"] = 0.35
tagger_args["rawtag"] = False # help='Use the raw output of the model'
tagger_args["recursive"] = False # help='Enable recursive file search')
tagger_args["model"] = 'wd-eva02-large-tagger-v3'
tagger_args["exclude_tags"] = None

# get interrogator configs
interrogator = interrogators[tagger_args["model"]]

def parse_exclude_tags() -> set[str]:
    if tagger_args["exclude_tags"] is None:
        return set()

    tags = []
    for str in tagger_args["exclude_tags"]:
        for tag in str.split(','):
            tags.append(tag.strip())

    # reverse escape (nai tag to danbooru tag)
    reverse_escaped_tags = []
    for tag in tags:
        tag = tag.replace(' ', '_').replace('\(', '(').replace('\)', ')')
        reverse_escaped_tags.append(tag)
    return set([*tags, *reverse_escaped_tags])  # reduce duplicates


def image_interrogate(image_path: Path, tag_escape: bool, exclude_tags: Iterable[str]) -> dict[str, float]:
    """
    Predictions from a image path
    """
    im = Image.open(image_path)
    result = interrogator.interrogate(im)

    return Interrogator.postprocess_tags(
        result[1],
        threshold=tagger_args["threshold"],
        escape_tag=tag_escape,
        replace_underscore=tag_escape,
        exclude_tags=exclude_tags)
    

# tags = image_interrogate(Path(args.file), not tagger_args["rawtag"], parse_exclude_tags())
# print()
# tags_str = ', '.join(tags.keys())
# print(tags_str)


