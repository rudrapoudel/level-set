----------------------------------------------------------------------
-- Distance Regularized Level Set Evolution (DRLSE)
--
-- Rudra Poudel
----------------------------------------------------------------------
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

-- Define sobel filters
-- local sobelx = torch.FloatTensor(3,3):fill(0)
-- sobelx[{1,1}] = 1; sobelx[{3,1}] = -1
-- sobelx[{1,2}] = 2; sobelx[{3,2}] = -2
-- sobelx[{1,3}] = 1; sobelx[{3,3}] = -1
-- sobelx:mul(1.0/4.0)
-- print(sobelx)

-- local sobely = torch.FloatTensor(3,3):fill(0)
-- sobely[{1,1}] = -1; sobely[{1,3}] = 1
-- sobely[{2,1}] = -2; sobely[{2,3}] = 2
-- sobely[{3,1}] = -1; sobely[{3,3}] = 1
-- sobely:mul(1.0/4.0)
-- print(sobely)

local sobelx = torch.FloatTensor(1,3):fill(0)
sobelx[{1,1}] = 0.5; sobelx[{1,3}] = -0.5
-- print(sobelx)

local sobely = torch.FloatTensor(3,1):fill(0)
sobely[{1,1}] = -0.5; sobely[{3,1}] = 0.5
-- print(sobely)

local laplace_op = torch.FloatTensor(3,3):fill(0)
laplace_op[{1,2}] = 1;
laplace_op[{2,1}] = 1; laplace_op[{2,2}] = -4; laplace_op[{2,3}] = 1
laplace_op[{3,2}] = 1;
-- print(laplace_op)

function getImageGrad(img)
    local gx, gy, h, w
    gx = image.convolve(img, sobelx, 'same')
    gy = image.convolve(img, sobely, 'same')

    -- Boundary cases
    w = gx:size(2)
    gx[{{},1}] = img[{{},2}] - img[{{},1}]
    gx[{{},w}] = img[{{},w}] - img[{{},w-1}]

    h = gx:size(1)
    gy[{1,{}}] = img[{1,{}}] - img[{2,{}}]
    gy[{h,{}}] = img[{h-1,{}}] - img[{h,{}}]  

    return gx, gy
end

function discreteLaplace(img)
    local l, h, w
    local l = image.convolve(img, laplace_op, 'same')

    -- boundary cases
    h = l:size(1)
    w = l:size(2)
    l[{{2,h-1},1}] = l[{{2,h-1},1}] + img[{{2,h-1},1}]
    l[{{2,h-1},w}] = l[{{2,h-1},w}] + img[{{2,h-1},w}]

    
    l[{1,{2,w-1}}] = l[{1,{2,w-1}}] + img[{1,{2,w-1}}]
    l[{h,{2,w-1}}] = l[{h,{2,w-1}}] + img[{h,{2,w-1}}]

    l[{1,1}] = l[{1,1}] + 2 * img[{1,1}]
    l[{1,w}] = l[{1,w}] + 2 * img[{1,w}]
    l[{h,1}] = l[{h,1}] + 2 * img[{h,1}]
    l[{h,w}] = l[{h,w}] + 2 * img[{h,w}]

    return l
end

function neumannBoundCond(f)
    local rows = f:size(1)
    local cols = f:size(2)

    f[{1,{}}] = f[{3,{}}]
    f[{rows,{}}] = f[{rows-2,{}}]

    f[{{},1}] = f[{{},3}] 
    f[{{},cols}] = f[{{},cols-2}] 

    f[{1,1}] = f[{3,3}] 
    f[{1,cols}] = f[{3,cols-2}]
    f[{rows,1}] = f[{rows-2,3}]
    f[{rows,cols}] = f[{rows-2,cols-2}]
    return f
end

function div(nx, ny)
    local nxx, nyy, temp
    nxx, temp = getImageGrad(nx)
    temp, nyy = getImageGrad(ny)
    return nxx:add(nyy)
end

function distRegularization(phi, pi)
    local phi_x, phi_y, s, sge, sle, a, b, ps, dps, f

    phi_x, phi_y = getImageGrad(phi)
    s = torch.add(torch.pow(phi_x,2), torch.pow(phi_y, 2)):sqrt()

    sge = torch.ge(s, 0):typeAs(phi)
    sle = torch.le(s, 1):typeAs(phi)
    a = torch.cmul(sge, sle):typeAs(phi)
    b = torch.gt(s, 1):typeAs(phi)

    ps = torch.mul(s, 2 * pi):sin():cmul(a):div(2 * pi):add( torch.add(s, -1):cmul(b))
    dps = torch.eq(ps, 0):typeAs(phi):add(ps):cdiv( torch.eq(s, 0):typeAs(phi):add(s) )
    f = div( torch.cmul(dps, phi_x):add(-1, phi_x), torch.cmul(dps, phi_y):add(-1, phi_y) )
    f:add( discreteLaplace(phi))

    return f
end

function dirac(x, sigma, pi)
    local pi_value, f, xge, xle, b
    pi_value = pi or 3.1416
    f = torch.mul(x, pi_value):div(sigma):cos():add(1):mul( 0.5/sigma)

    xge = torch.ge(x, -sigma):typeAs(x)
    xle = torch.le(x, sigma):typeAs(x)
    b = torch.cmul(xge, xle):typeAs(x)
    f:cmul(b)

    return f
end

function drlse(phi0, g, params, shape_prior)
    local phi, dphi, vx, vy, phi_x, phi_y, s, Nx, Ny, curvature 
    local dist_regularization, direc_phi, area_term, edge_term
    
    phi = phi0
    vx, vy = getImageGrad(g)
    for k=1, params.iter_inner do
        phi = neumannBoundCond(phi)
        phi_x, phi_y = getImageGrad(phi)
        s = torch.add(torch.pow(phi_x,2), torch.pow(phi_y, 2)):sqrt():add(1e-10)
        Nx = torch.cdiv(phi_x, s)
        Ny = torch.cdiv(phi_y, s)

        curvature = div(Nx, Ny)

        if params.fn_potential == 'single-well' then
            dist_regularization = discreteLaplace(phi)
            dist_regularization:add(-1, curvature)
        elseif params.fn_potential == 'double-well' then
            dist_regularization = distRegularization(phi, params.pi)
        else
            error('UNKNOWN fn_potential type: ' .. params.fn_potential)
        end

        direc_phi = dirac(phi, params.epsilon, params.pi)

        area_term = torch.cmul(direc_phi, g)    -- balloon/pressure force
        edge_term = torch.cmul(vx, Nx):add(torch.cmul(vy, Ny)):cmul(direc_phi)
        edge_term:add( torch.cmul(area_term, curvature))

        dphi = torch.mul(dist_regularization, params.mu)
        dphi:add(params.lambda, edge_term):add(params.alfa, area_term):mul(params.time_step)

        -- Add shape prior
        if params.beta > 0 then
            dphi:add( torch.add(phi, -1, shape_prior):mul(params.beta) )
        end
        phi:add(dphi)
    end
end

-- img and phi0 will be modified
function applyDRLSE(img, phi0, drlse_params, gkernel, shape_prior)
    local simg, gx, gy, f, seg_img
    simg = image.convolve(img, gkernel, 'same')
    
    -- Edge indicator
    gx, gy = getImageGrad(simg)
    f = torch.add(gx:pow(2), gy:pow(2))
    f:add(1):pow(-1) -- 1/(1+f)

    -- DRLSE outer loop
    for i=1, drlse_params.iter_outer do
        drlse(phi0, f, drlse_params, shape_prior)
    end

    -- Refine the zero level contour by further level set evolution with alfa=0
    drlse_params.alfa = 0
    drlse(phi0, f, drlse_params, shape_prior)

    -- print(phi0)
    seg_img = torch.le(phi0, 0):typeAs(simg)

    return seg_img
end

function test_drlse()
    local log, img, shape_prior, phi0, gkernel, seg_img
    
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

    -- Init phi 0
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
    
    img:mul(255)
    seg_img = applyDRLSE(img, phi0, drlse_params, gkernel, shape_prior)

    -- DEBUG
    image.save(log .. 'seg_img.png', seg_img)
end


--------------------------------
-- main()

-- Test GRAD
-- local xx = torch.FloatTensor(35)
-- for i=1,35 do
--     xx[i] = i
-- end
-- local x = xx:view(7, 5)
-- print(x)
-- local gx, gy = getImageGrad(x)
-- print(gx)
-- print(gy)


-- Test LAPLACE
-- local aa = {3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
-- local xx = torch.FloatTensor(16)
-- for i=1,16 do
--     xx[i] = aa[i]
-- end
-- local x = xx:view(4, 4)
-- print(x)
-- local l = discreteLaplace(x)
-- print(l)


-- Test DRLSE
test_drlse()
--error('BOOM BOOM BOOM')
