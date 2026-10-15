# 2D Haar downsampling: returns LL (approx) and the three detail maps
# Input: (H, W, C, B) — assumes even H, W
function haar_down(x::AbstractArray)
    H, W, C, B = size(x)
    # Vertical then horizontal averaging/differencing
    # Even/odd rows
    x_even_r = x[1:2:H, :, :, :]
    x_odd_r  = x[2:2:H, :, :, :]
    # Horizontal
    # x_even_c = x[:, 1:2:W, :, :]
    # x_odd_c  = x[:, 2:2:W, :, :]

    # LL, LH, HL, HH (normalized by 1/√2 or 1/2 for convenience; scaling is arbitrary for learning)
    LL = (x_even_r[:, 1:2:W, :, :] + x_even_r[:, 2:2:W, :, :] +
          x_odd_r[:, 1:2:W, :, :]  + x_odd_r[:, 2:2:W, :, :]) ./ 2
    LH = (x_even_r[:, 1:2:W, :, :] - x_even_r[:, 2:2:W, :, :] +
          x_odd_r[:, 1:2:W, :, :]  - x_odd_r[:, 2:2:W, :, :]) ./ 2
    HL = (x_even_r[:, 1:2:W, :, :] + x_even_r[:, 2:2:W, :, :] -
          x_odd_r[:, 1:2:W, :, :]  - x_odd_r[:, 2:2:W, :, :]) ./ 2
    HH = (x_even_r[:, 1:2:W, :, :] - x_even_r[:, 2:2:W, :, :] -
          x_odd_r[:, 1:2:W, :, :]  + x_odd_r[:, 2:2:W, :, :]) ./ 2
    return LL, LH, HL, HH
end

haar_down_ll(x::AbstractArray) = haar_down(x)[1]


HaarCB(ch_in, ch_out, activation) = 
    Chain(
        # MaxPool((2,2); stride=2),
        x->haar_down_ll(x),
        ConvK3(ch_in, ch_out, activation),
        ConvK3(ch_out, ch_out),
        BatchNorm(ch_out, activation),
)


# Simple inverse (for optional reconstruction path or testing)
function haar_up(LL, LH, HL, HH)
    # Reconstruct even/odd rows and columns (inverse of the operations above)
    # This is a minimal version; full reconstruction is straightforward linear combinations
    H2, W2, C, B = size(LL)
    # ... implement the four combinations that recover the original even/odd positions
    # (omitted for brevity — see numerical-tours Haar 2D for the exact formulas)
    # For training, you often only need the forward path + concatenation of details
end