"""Open Images V7: a random sample loaded with the FiftyOne zoo, and labels for coloring maps."""

from .config import SEED, ZOO_DIR


def load_dataset(split="validation", max_samples=25000, zoo_dir=ZOO_DIR, name=None, seed=SEED):
    """Load a persistent FiftyOne dataset of `max_samples` random images of `split`, downloading them if needed.

    Only image-level labels are loaded (field `positive_labels`, a Classifications), plus the class hierarchy
    in `dataset.info["hierarchy"]`. The name includes split and size (e.g. open-images-v7-validation-25k):
    an existing dataset is loaded as is, so another size is another dataset (images already downloaded are reused).
    Files go to `zoo_dir/open-images-v7/<split>/`.
    """
    import fiftyone as fo
    import fiftyone.zoo as foz

    name = name or f"open-images-v7-{split}-{max_samples // 1000}k"
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    fo.config.dataset_zoo_dir = str(zoo_dir)
    dataset = foz.load_zoo_dataset(
        "open-images-v7", split=split, label_types=["classifications"], max_samples=max_samples,
        shuffle=True, seed=seed, dataset_name=name,
    )
    dataset.persistent = True
    return dataset


def image_labels(dataset, field="positive_labels"):
    """The positive class names of every sample, e.g. [["Dog", "Mammal", "Animal", ...], ...]."""
    return [[c.label for c in cls.classifications] if cls else [] for cls in dataset.values(field)]


def descendants(hierarchy, name):
    """`name` and every class below it in the hierarchy (only the 600 boxable classes have one)."""
    def find(node):
        if node.get("LabelName") == name:
            return node
        for child in node.get("Subcategory", []):
            if (found := find(child)) is not None:
                return found
        return None

    def collect(node):
        yield node["LabelName"]
        for child in node.get("Subcategory", []):
            yield from collect(child)

    node = find(hierarchy)
    if node is None:
        raise KeyError(f"{name!r} is not in the hierarchy")
    return set(collect(node))


def group_labels(labels_per_sample, hierarchy, groups, other="other", ignore=()):
    """One label per sample: the first of `groups` (hierarchy classes, e.g. "Animal") it has a class under, else `other`.

    Also works for specific classes (e.g. "Bird", which then includes "Eagle"). Classes in `ignore` don't count for any group. Annotators label people as "Mammal" (under "Animal"), so pass
    `ignore={"Mammal"}` to keep photos of people out of "Animal".
    Labels are independent of the embeddings, so coloring by them shows what the embedding groups together.
    """
    members = {g: descendants(hierarchy, g) - set(ignore) for g in groups}
    return [next((g for g in groups if members[g] & set(labels)), other) for labels in labels_per_sample]
