"""Create byte-bounded JPEG derivatives; source bytes are never overwritten."""

import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageToolError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def open_image(data, config):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            picture = Image.open(io.BytesIO(data))
            if picture.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ImageToolError('unsupported_image', 'Use a PNG, JPEG or WebP still image.')
            if picture.width * picture.height > config.max_pixels:
                raise ImageToolError('image_too_many_pixels', 'Image exceeds the configured pixel limit.')
            if getattr(picture, 'n_frames', 1) != 1:
                raise ImageToolError('animated_image', 'Animated images are not supported; select a still frame.')
            picture.load()
            return ImageOps.exif_transpose(picture)
    except ImageToolError:
        raise
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageToolError('invalid_image', 'Cannot decode this image within the configured limits.') from exc


def describe_image(data, config):
    with open_image(data, config) as picture:
        has_alpha = 'A' in picture.getbands() or 'transparency' in picture.info
        transparent = has_alpha and picture.convert('RGBA').getchannel('A').getextrema()[0] < 255
        return {'width': picture.width, 'height': picture.height,
                'has_alpha': has_alpha, 'has_transparency': bool(transparent)}


def make_preview(data, config, crop=None, background='white'):
    if background not in ('white', 'checkerboard'):
        raise ImageToolError('invalid_background', 'background must be white or checkerboard.')
    with open_image(data, config) as original:
        if crop is not None:
            if (not isinstance(crop, (list, tuple)) or len(crop) != 4
                    or any(type(n) is not int for n in crop)):
                raise ImageToolError('invalid_crop', 'crop must contain four integer pixel coordinates.')
            left, top, right, bottom = crop
            if not (0 <= left < right <= original.width and 0 <= top < bottom <= original.height):
                raise ImageToolError('invalid_crop', 'crop must be inside the oriented source image.')
            picture = original.crop(crop)
        else:
            picture = original.copy()
    picture.thumbnail((config.preview_max_edge, config.preview_max_edge), Image.Resampling.LANCZOS)
    rgba = picture.convert('RGBA')
    rgb = Image.new('RGB', rgba.size, 'white')
    if background == 'checkerboard':
        from PIL import ImageDraw
        draw = ImageDraw.Draw(rgb)
        for y in range(0, rgb.height, 16):
            for x in range(0, rgb.width, 16):
                if (x // 16 + y // 16) % 2:
                    draw.rectangle((x, y, x + 15, y + 15), fill='#d4d4d4')
    rgb.paste(rgba, mask=rgba.getchannel('A'))
    # Strip metadata and EXIF, and never fall back to returning the original.
    while True:
        for quality in (82, 72, 60, 48):
            target = io.BytesIO()
            rgb.save(target, format='JPEG', quality=quality, optimize=True)
            encoded = target.getvalue()
            if len(encoded) <= config.preview_max_bytes:
                return encoded, {'width': rgb.width, 'height': rgb.height,
                                 'bytes': len(encoded), 'mime_type': 'image/jpeg',
                                 'background': background, 'crop': crop,
                                 'transparency_evidence': False}
        next_edge = int(max(rgb.size) * 0.8)
        if next_edge < 320:
            raise ImageToolError('preview_budget_exceeded', 'Choose a smaller crop; cannot make a readable preview within budget.')
        rgb.thumbnail((next_edge, next_edge), Image.Resampling.LANCZOS)
