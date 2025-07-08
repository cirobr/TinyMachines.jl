# fpos(x) = ifelse(x>0, x, zero(x))   # fpos(x) = max(x, 0)
# fneg(x) = ifelse(x<0, x, zero(x))   # fneg(x) = min(x, 0)
# fneg(x) = x .- relu(x)
const fpos = relu
fneg(x) = ifelse(x>0, zero(x), x)  # fneg(x) = min(x, 0)
