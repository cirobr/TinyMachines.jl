fpos(x) = ifelse(x>0, x, zero(x))   # fpos(x) = max(x, 0)
fneg(x) = ifelse(x<0, x, zero(x))   # fneg(x) = min(x, 0)
