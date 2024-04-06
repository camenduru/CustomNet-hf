from ldm.util import load_and_preprocess
from carvekit.api.high import HiInterface
import spaces


def load_preprocess_model():
    carvekit = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        # device='cuda' if torch.cuda.is_available() else 'cpu',
                        device='cuda',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)
    return carvekit
    
@spaces.GPU
def preprocess_image(models, input_im):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array.
    '''
    input_im = load_and_preprocess(models, input_im)
    return input_im