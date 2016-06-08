----------------------------------------------------------------------
-- Otsu threshold algorithm for the binary images
--
-- Rudra Poudel
----------------------------------------------------------------------
require 'torch'
require 'image'

local otsu = torch.class('OtsuThreshold')

function otsu:__init()
    self.bins = 256
    self.hist = torch.FloatTensor(self.bins)
    self.threshold = -1
    self.max_level_value = 0
end

function otsu:getThreshold()
    assert(self.threshold>=0)
    return self.threshold
end

function otsu:doThreshold(src, dst)
    assert(src:min() >= 0)
    assert(src:max() <= 255)
    local bin_id, rows, cols
    local sum, sumB, wB, wF, mB, mF, threshold1, threshold2, total, var_between, var_max

    self.hist:fill(0)
    rows = src:size(1)
    cols = src:size(2)
    for row=1,rows do
        for col=1,cols do
            bin_id = src[{row, col}] + 1
            self.hist[bin_id] = self.hist[bin_id] + 1

            -- if self.hist[bin_id] > self.max_level_value then
            --     self.max_level_value = self.hist[bin_id]
            -- end
        end
    end

    sum = 0.0
    for b=1,self.bins do sum = sum + ((b-1) * self.hist[b]) end
    sumB = 0.0
    wB = 0.0
    wF = 0.0

    var_max = 0.0
    threshold1 = 0
    threshold2 = 0
    total = rows * cols
    for b=1,self.bins do
        wB = wB + self.hist[b]
        if wB > 0 then
            wF = total - wB
            if wF == 0 then break end

            sumB = sumB + ((b-1) * self.hist[b])
            mB = sumB / wB
            mF = (sum - sumB)/wF

            -- Calculate between class variance
            var_between = wB * wF * (mB - mF) * (mB - mF)
            if (var_between >= var_max) then
                threshold1 = b-1
                if (var_between > var_max) then
                    threshold2 = b-1
                end
                var_max = var_between
            end
        end
    end
    self.threshold = ( threshold1 + threshold2) / 2.0

    if dst then
        gray = torch.ge(dst, self.threshold):typeAs(dst)
        return gray
    end
end

function testOtsuThreshold()
    local log, img, gray
    
    log = 'log/'

    img = image.load('image/house.jpg')[1]
    -- DEBUG
    -- image.save(log .. 'otsu_in.png', img)

    img:mul(255):round()
    otsu = OtsuThreshold()
    gray = otsu:doThreshold(img, img)

    -- DEBUG
    if gray then
        image.save(log .. 'otsu_out_lua.png', gray)
        print(otsu:getThreshold())
    end
end


-- MAIN() -- for testing
testOtsuThreshold()