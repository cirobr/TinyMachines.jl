@info "miscellaneous tests..."

@test tm.fpos(4) == 4   || error("fpos(4) should equal 4.")
@test tm.fpos(-4) == 0  || error("fpos(-4) should equal 0.")
@test tm.fneg(4) == 0   || error("fneg(4) should equal 0.")
@test tm.fneg(-4) == -4 || error("fneg(-4) should equal -4.")
@test tm.fpos(0) == 0   || error("fpos(0) should equal 0.")
@test tm.fneg(0) == 0   || error("fneg(0) should equal 0.")