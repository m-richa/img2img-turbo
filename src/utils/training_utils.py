import os, random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


def parse_args_paired_training(input_args=None):
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=10, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)
    parser.add_argument("--lambda_l1", default=2.0, type=float)
    parser.add_argument("--lambda_clipsim", default=0.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=1, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="linear",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--face_disc", default=False, action="store_true")
    parser.add_argument("--warping", default=False, action="store_true")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def build_transform(image_prep):
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    return T



class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer
        self.gt_frame_path = "/NAS5/speech/user/richamishra/paii_virtual_being_3d/img2img-turbo/data/jo/train_B/00001.png"
        
    def face_crop(self, image, is_output=False):
        
        _, h, w = image.shape
        
        if is_output:
            image = (image+1.0)/2.0 #[0,1]
            
        face_crop = F.crop(image, 0, 128, 200, 200)
        face_crop = F.interpolate(face_crop, size = [512, 512], mode='bilinear', align_corners=True)
        
        if is_output:
            face_crop = (face_crop*2.0) -1.0
            
        return face_crop
        
        
    def crop(self, image, i=None, j=None, is_output=False):
        
        """
        image = [0,1] --> crop --> [0, 1]
        """
        _, h, w = image.shape
    
        if is_output:
            image = (image + 1.0)/2.0
        
        if i==None:
            i = random.randint(-50, 0)
        if j==None:
            j = random.randint(-50, 50)
            
        cropped_image = F.crop(image, i, j, h, w)
        
        if is_output:
            cropped_image = (cropped_image*2.0) - 1.0
        
        return cropped_image, i, j
        

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):

        img_name = self.img_names[idx]
        gt_frame = Image.open(self.gt_frame_path)
        #gt_frame_kp = Image.open(self.gt_frame_path.replace('train_B', 'train_A'))
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]
        
        # input images scaled to 0,1
        
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        #mask = img_t.sum(0)
        #mask = torch.where(mask==3, 0, 1).unsqueeze(0) #bg is 0
        #img_t = img_t*mask#make the bg = 0
        
        img_t, i, j = self.crop(img_t)
        
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        mask_output = output_t[3,:,:]
        mask_output = torch.where(mask_output>0, 1.0, 0.0)
        
        output_t = output_t[:3,:,:]
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])
        output_t,_,_ = self.crop(output_t, i, j, True)
        mask_output,_,_ = self.crop(mask_output.unsqueeze(0), i, j)
        
        """"""
        gt_frame = self.T(gt_frame)
        gt_frame = F.to_tensor(gt_frame)
        gt_frame = gt_frame[:3,:,:]
        
        # gt_frame_kp = self.T(gt_frame_kp)
        # gt_frame_kp = F.to_tensor(gt_frame_kp)
        # gt_frame_kp = gt_frame_kp[:3,:,:]
        # gt_frame_kp = F.normalize(gt_frame_kp, mean=[0.5], std=[0.5])
        
        # gt_frame = torch.cat([gt_frame, gt_frame_kp], dim=0)
        """"""

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "gt_frame": gt_frame,
            "conditioning_pixel_values": img_t,
            "mask_output": mask_output,
            "crop": [i,j],
            "caption": caption,
            "input_ids": input_ids,
        }
