# COCO-style run-length encoded binary mask.
#
# `counts` holds alternating run lengths starting with a run of zeros
# (background); `size` is [height, width] in pixels. pycocotools can decode
# directly. Empty `counts` (with status_code != 0) indicates "no mask
# available" — downstream should fall back to bbox or abort.
#
# This message intentionally carries no category / class label. sam_manager
# does not classify; if a single channel of a multi-class backend output is
# being returned, the consumer chose which class it asked about via the
# reference image+mask pair.

int32[] counts
int32[] size
