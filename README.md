##Level Set

Implementation of distance regularized level-set evolution (DRLSE) [[1]](#references) in lua.

## Usage

```
    local log, img, shape_prior, phi0, gkernel, seg_img
    
    -- Load image
    log = 'log/'
    img = image.load('image/gourd.jpg')[1]
    shape_prior = image.load('image/shape_prior.png')[1]

    -- Set DRLSE params
    local drlse_params = {}
    drlse_params.time_step = 1
    drlse_params.iter_inner = 5;
    drlse_params.iter_outer = 20;
    drlse_params.gauss_filter_size = 5
    drlse_params.gauss_sigma = 0.8    -- scale parameter in Gaussian kernel
    drlse_params.pi = 3.1416

    drlse_params.phi_c0 = 2
    drlse_params.fn_potential = 'double-well' --'single-well' | 'double-well'
    drlse_params.mu = 0.2/drlse_params.time_step  -- coefficient of the distance regularization term R(phi)
    drlse_params.lambda = 5    -- coefficient of the weighted length term L(phi)
    drlse_params.alfa = -3     -- coefficient of the weighted area term A(phi)
    drlse_params.epsilon = 1.5 -- papramater that specifies the width of the DiracDelta function
    drlse_params.beta = 0 --0.02     -- coefficient of the shape prior of the LV

    -- Smooth the image
    gkernel = image.gaussian{size = drlse_params.gauss_filter_size, sigma = drlse_params.gauss_sigma}
    assert(gkernel:sum()>1)
    gkernel:mul(1.0/gkernel:sum())

    -- Init phi
    phi0 = img:clone()
    phi0:fill(drlse_params.phi_c0)

    -- Init phi 0- i.e. starting point of the level-set
    for i=25, 28 do
        for j=20, 23 do
            phi0[{i,j}] = -drlse_params.phi_c0
        end
    end
    for i=25, 28 do
        for j=40, 43 do
            phi0[{i,j}] = -drlse_params.phi_c0
        end
    end
    
    -- Call main DRLSE level-set function
    img:mul(255)
    seg_img = applyDRLSE(img, phi0, drlse_params, gkernel, shape_prior)

    -- DEBUG- save image to view the result
    image.save(log .. 'seg_img.png', seg_img)
```

##Ouput Examples,

An example for a synthetic image
![Alt text](image/gourd_contour.gif?raw=true "DRLSE Example")

An example for a real cell image
![Alt text](image/twocells_contour.gif?raw=true "DRLSE Example")

##Otsu Threshold

otsu_threshold.py: implementation of Otsu threshold in lua. Other two related files are for cross-validation from the python skimage implementation.

## References

[1] Li, C., Xu, C., Gui, C., Fox, M.D., 2010. Distance Regularized Level Set Evolution and Its Application to Image Segmentation. IEEE Transactions on Image Processing 19, 3243–3254.