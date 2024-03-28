TextEncoders = dict()
TextEncoders["bert"] = dict(
    name="bert_base",
    pretrained="/mnt/share_disk/zhangguiwei/bert-base-uncased",
    config="configs/config_bert.json",
    d_model=768,
    fusion_layer=9,
)
TextEncoders["bert_large"] = dict(
    name="bert_large",
    pretrained="/mnt/share_disk/zhangguiwei/bert-large-uncased",
    config="configs/config_bert_large.json",
    d_model=1024,
    fusion_layer=19,
)
