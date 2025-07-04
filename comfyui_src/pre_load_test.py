
from func import Expression_difference, i2i, t2i, unload_pipe


for i in range(2):
    i2i(
    ref_img_path='input/20201221183626_c9342.jpg',
    style_name="style39",
    positive_prompt="1girl,white dress,holding flower,",
    negative_prompt="",
    target_width=1024,
    target_height=1536,
    save_img=True)

# for i in range(40):
#     t2i(style_name="style39",
#     positive_prompt="1girl,white dress,holding flower,",
#     negative_prompt="",
#     target_width=1024,
#     target_height=1536,
#     save_img=True)

# for i in range(40):
#     i2i(
#     ref_img_path='input/20201221183626_c9342.jpg',
#     style_name="style38-no-deep",
#     positive_prompt="1girl,white dress,holding flower,",
#     negative_prompt="",
#     target_width=1024,
#     target_height=1536,
#     save_img=True)

# for i in range(40):
#     t2i(style_name="style38-no-deep",
#     positive_prompt="1girl,white dress,holding flower,",
#     negative_prompt="",
#     target_width=1024,
#     target_height=1536,
#     save_img=True)

# scp -r wtz@35.185.237.104:/home/wtz/diffuser-pipe/src/app/ComfyUI/output output5
