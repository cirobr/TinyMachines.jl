fpos(x) = ifelse(x<0, zero(x), x)  # same implementation as relu
fneg(x) = ifelse(x>0, zero(x), x)
