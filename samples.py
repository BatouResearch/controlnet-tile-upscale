import base64
import requests
import sys
import os


def gen(output_fn, **kwargs):
    if os.path.exists(output_fn):
        print("Skipping", output_fn)
        return

    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)

    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    gen(
        "sample.none.png",
        prompt="taylor swift in a mid century modern bedroom",
        seed=42,
        steps=30,
    )
    gen(
        "sample.canny.png",
        prompt="taylor swift in a mid century modern bedroom",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.qr.png",
        prompt="A film still of a kraken, reconciliation, 8mm film, traditional color grading, cinemascope, set in 1878",
        qr_image="https://github.com/anotherjesse/dream-templates/assets/27/c5df2f7c-7a0c-43ad-93d6-921af0759190",
        qr_conditioning_scale=1.5,
        seed=42,
        scheduler="K_EULER",
        steps=50,
    )
    gen(
        "sample.canny.guess.png",
        prompt="",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.hough.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.hough.guess.png",
        prompt="",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.normal.png",
        prompt="",
        normal_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.depth.png",
        prompt="",
        depth_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.both.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        seed=42,
        steps=30,
    )
    gen(
        "sample.both.guess.png",
        prompt="",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        guess_mode=True,
        seed=42,
        steps=30,
    )
    gen(
        "sample.scaled.png",
        prompt="taylor swift in a mid century modern bedroom",
        hough_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        hough_conditioning_scale=0.6,
        canny_image="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        canny_conditioning_scale=0.8,
        seed=42,
        steps=30,
    )
    gen(
        "sample.seg.png",
        prompt="modern bedroom with plants",
        seg_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
    )
    gen(
        "sample.hed.png",
        prompt="modern bedroom with plants",
        hed_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/room_512x512.png",
        seed=42,
    )
    gen(
        "sample.pose.png",
        prompt="a man in a suit by van gogh",
        pose_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/human_512x512.png",
        seed=42,
    )
    gen(
        "sample.scribble.png",
        prompt="painting of cjw by van gogh",
        scribble_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/converted/control_vermeer_scribble.png",
        seed=42,
    )



if __name__ == "__main__":
    main()
