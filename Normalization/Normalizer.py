import torch


class Normalizer:
    @classmethod
    def get_bounding_box_normalizer(cls, x: torch.Tensor):
        """A method to get a normalizer that normalizes the input tensor to a unit cube.

        Args:
            x (torch.Tensor): The input tensor to normalize. (N, D) where N is the number of points and D is the dimension of each point.

        Returns:
            Normalizer: An instance of Normalizer that can normalize the input tensor to a unit cube.
        """
        # Calculate the shift as the mean of the input tensor along the first dimension
        shift = torch.mean(x, dim=0)

        # Calculate the scale as the maximum L1 norm distance from the shift
        scale = torch.max(torch.norm(x - shift, p=1, dim=1))

        # Return an instance of Normalizer with the calculated scale and shift
        return Normalizer(scale=scale, shift=shift)

    @classmethod
    def get_bounding_sphere_normalizer(cls, x: torch.Tensor):
        """A method to get a normalizer that normalizes the input tensor to a unit sphere.

        Args:
            x (torch.Tensor): The input tensor to normalize. (N, D) where N is the number of points and D is the dimension of each point.

        Returns:
            Normalizer: An instance of Normalizer that can normalize the input tensor to a unit sphere.
        """
        # Calculate the shift as the mean of the input tensor along the first dimension
        shift = torch.mean(x, dim=0)

        # Calculate the scale as the maximum L2 norm distance from the shift
        scale = torch.max(torch.norm(x - shift, p=2, dim=1))

        # Return an instance of Normalizer with the calculated scale and shift
        return Normalizer(scale=scale, shift=shift)

    def __init__(self, scale: torch.Tensor, shift: torch.Tensor):
        """A constructor for the Normalizer class.

        Args:
            scale (torch.Tensor): The scale vector to normalize the input tensor.
            shift (torch.Tensor): The shift vector to normalize the input tensor.
        """
        self._scale = scale
        self._shift = shift

    def __call__(self, x: torch.Tensor):
        """A method to normalize the input tensor.

        Args:
            x (torch.Tensor): The input tensor to normalize. (N, D) where N is the number of points and D is the dimension of each point.

        Returns:
            torch.Tensor: The normalized input tensor.
        """
        return (x - self._shift) / self._scale

    def get_de_normalizer(self):
        """A method to get a de-normalizer that can de-normalize the input tensor.

        Returns:
            Normalizer: An instance of Normalizer that can de-normalize the input tensor.
        """
        # Calculate the inverse scale and shift
        inv_scale = 1 / self._scale
        inv_shift = -self._shift / self._scale
        return Normalizer(scale=inv_scale, shift=inv_shift)
