import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class PredictorVisualizer_S:
    def __init__(self, model, checkpoint_path, dataset, device='cuda:7',cad_type='asap7'):
        """
        Initializes the PredictorVisualizer with the model, checkpoint, dataset, and device.
        
        Args:
            model (torch.nn.Module): The PyTorch model to be used for predictions.
            checkpoint_path (str): Path to the model checkpoint.
            dataset (torch.utils.data.Dataset): The dataset to use for predictions.
            device (str): The device to use for computation ('cuda:X' or 'cpu').
        """
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.device = device
        self.cad_type = cad_type
        self._load_model()

    def _load_model(self):
        """Loads the model weights from the checkpoint."""
        self.model.load_state_dict(torch.load(self.checkpoint_path)['net'])
        self.model.to(self.device)
        self.model.eval()

    def compute_saliency_maps(self, input_tensor, target):
        """
        Computes saliency maps for each channel of the input tensor.
        
        Args:
            input_tensor (torch.Tensor): Input tensor to compute saliency for.
            target (torch.Tensor): Target tensor for loss computation.
            
        Returns:
            torch.Tensor: Saliency maps for each input channel.
        """
        # Create a copy of input tensor that requires gradients
        input_tensor = input_tensor.clone().detach().to(self.device)
        input_tensor.requires_grad_()
        target = target.to(self.device)


        # Forward pass
        output = self.model(input_tensor)
        threshold = torch.quantile(target, 0.9)
        high_drop_mask = (target >= threshold).float().unsqueeze(0)

        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=high_drop_mask)
        
        # Get gradients
        saliency_map = input_tensor.grad.abs()
       
     
        return self.min_max_norm(saliency_map)

    def predict(self, index, norm_out=False):
        """
        Makes a prediction for the given index from the dataset.
        
        Args:
            index (int): Index of the sample in the dataset.
            norm_out (bool): Whether to normalize the output using min-max normalization.
        
        Returns:
            tuple: (input, target, prediction, mae_map, saliency_maps)
        """
        sample = self.dataset.__getitem__(index)
        inp, target = sample[0], sample[1]
        
        # Prepare input tensor (add batch dimension)
        inp_batch = inp.unsqueeze(0).to(self.device)
        target_batch = target.unsqueeze(0).to(self.device)
        
        # Compute saliency maps
        saliency_maps = self.compute_saliency_maps(inp_batch, target_batch)
        
        # Perform prediction
        with torch.no_grad():
            pred = self.model(inp_batch)
        pred = pred.detach().cpu()

        # Normalize prediction if needed
        if norm_out:
            pred = self.min_max_norm(pred)

        # Calculate MAE map
        mae_map = torch.abs(pred - target)
        print('mae : ', torch.mean(mae_map))
        
        return inp, target, pred.squeeze(0), mae_map, saliency_maps.squeeze(0).detach().cpu()

    def process_input_images(self, inp, saliency_maps=None, saliency_map=False, saliency_map_background=False):
        """
        Processes input images based on saliency map options.
        
        Args:
            inp (torch.Tensor): Input tensor.
            saliency_maps (torch.Tensor): Saliency maps for each channel.
            saliency_map (bool): Whether to show saliency maps.
            saliency_map_background (bool): Whether to overlay saliency maps on input.
            
        Returns:
            list: Processed images for visualization.
        """

        if not saliency_map:
            return [inp] if inp.dim() == 2 else [inp[i] for i in range(inp.shape[0])]
            
        if saliency_maps is None:
            raise ValueError("Saliency maps required when saliency_map=True")
            
        processed_images = []
        for i in range(inp.shape[0]):  # For each channel
            if saliency_map_background:
                # Normalize both input and saliency map to [0,1]
                norm_inp = self.min_max_norm(inp[i])
                norm_saliency = self.min_max_norm(saliency_maps[i])
                # Overlay saliency map on input (alpha blending)
                processed_images.append(0.7 * norm_inp + 0.3 * norm_saliency)
            else:
                # Use saliency map directly
                processed_images.append(saliency_maps[i])
                
        return processed_images

    def visualize(self, inp, target, pred, mae_map, saliency_maps=None, cols=4, colorbar=False, 
                 cmap='inferno', saliency_map=False, saliency_map_background=False):
        """
        Visualizes the input, target, prediction, and MAE map.
        
        Args:
            inp (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
            pred (torch.Tensor): The predicted tensor.
            mae_map (torch.Tensor): The MAE map tensor.
            saliency_maps (torch.Tensor): Saliency maps for each channel.
            cols (int): Number of columns in the plot grid.
            colorbar (bool): Whether to include colorbars in the plots.
            cmap (str): Colormap to use for visualization.
            saliency_map (bool): Whether to show saliency maps.
            saliency_map_background (bool): Whether to overlay saliency maps on input.
        """
        cad_titles_common = ['Current', 'Effective Distance', 'PDN Density']
        cad_titles_asap7 = ['M2', 'M5', 'M6', 'M7', 'M8', 'M25', 'M56', 'M67', 'M78']
        cad_titles_nangate45 = ['M1', 'M4', 'M7', 'M8', 'M9', 'M14', 'M47', 'M78', 'M89']

        # Get titles based on cad_type
        if self.cad_type == 'asap7':
            input_titles = cad_titles_common + cad_titles_asap7
        elif self.cad_type == 'nangate45':
            input_titles = cad_titles_common + cad_titles_nangate45
        else:
            raise ValueError(f"Unsupported CAD type: {self.cad_type}")
        
        # Process input images based on saliency map options
        all_images = self.process_input_images(inp, saliency_maps, saliency_map, saliency_map_background)
        all_images += [target, pred, mae_map]

        # Calculate number of rows
        rows = (len(all_images) + cols - 1) // cols

        # Create plots
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten() if rows > 1 else [axes]

        image_type = 'Saliency' if saliency_map and not saliency_map_background else 'Input'
        titles = input_titles[:len(all_images) - 3] + ['Target', 'Prediction', 'MAE Map']

        for i, (img, title) in enumerate(zip(all_images, titles)):
            img = img.squeeze().cpu().numpy()
            ax = axes[i]
            if (colorbar and i >= len(titles) - 3) or self.input_colorbar:
                kwargs = {'vmin':0, 'vmax':1}
            else:
                kwargs = {}
            im = ax.imshow(img, cmap=cmap, **kwargs) 
            ax.set_title(title)
            ax.axis('off')
            if colorbar and (i > len(titles) - 4 or self.input_colorbar):
                fig.colorbar(im, ax=ax)

        # Remove unused subplots
        for i in range(len(all_images), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def predict_and_visualize(self, index, cols=4, colorbar=False, norm_out=False, 
                            casename=False, cmap='inferno', saliency_map=False, 
                            saliency_map_background=False,input_colorbar=False):
        """
        Combines prediction and visualization for a given index.
        
        Args:
            index (int): Index of the sample in the dataset.
            cols (int): Number of columns in the plot grid.
            colorbar (bool): Whether to include colorbars in the plots.
            norm_out (bool): Whether to normalize the output using min-max normalization.
            casename (bool): Whether to print the case name (if available).
            cmap (str): Colormap to use for visualization.
            saliency_map (bool): Whether to show saliency maps.
            saliency_map_background (bool): Whether to overlay saliency maps on input.
        """
        self.input_colorbar=input_colorbar
        inp, target, pred, mae_map, saliency_maps = self.predict(index, norm_out)
        
        if casename and len(self.dataset[index]) > 2:
            print(self.dataset[index][2])
        
        # Calculate metrics
        self.calculate_metrics(pred, target)
        
        # Visualize results
        self.visualize(inp, target, pred, mae_map, saliency_maps, 
                      cols, colorbar, cmap, saliency_map, saliency_map_background)

    @staticmethod
    def min_max_norm(tensor):
        """Applies min-max normalization to the given tensor."""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val == 0:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)

    def calculate_metrics(self, pred, target, th=90):
        pred_np = to_numpy(pred)
        target_np = to_numpy(target)
        
        # Threshold using percentile
        target_np = (target_np >= np.percentile(target_np, th)).astype(int)
        pred_np = (pred_np >= np.percentile(pred_np, th)).astype(int)
     
        # F1 score
        f1=dice_coeff = f1_score(target_np.flatten(), pred_np.flatten())
        
        
        print(f"Dice Coefficient: {dice_coeff:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        return dice_coeff, f1


def to_numpy(x):
    return x.detach().cpu().numpy()