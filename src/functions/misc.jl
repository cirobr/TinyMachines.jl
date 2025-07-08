oftf(x, y) = oftype(float(x), y)   # convert y to the type of float(x)
fpos(x) = ifelse(x<0, zero(x), x)  # same implementation as relu
fneg(x) = ifelse(x>0, zero(x), x)
