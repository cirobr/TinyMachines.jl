using TinyMachines; tm= TinyMachines
using Flux

# Test the ESPmodule
ch_in, ch_out = 3, 10
esp = tm.ESPmodule(ch_in, ch_out; K=5, add=false)
x = randn(Float32, (32, 32, ch_in, 1))
y = esp(x)
size(y), typeof(y)

ch_in, ch_out = 10, 10
esp = tm.ESPmodule(ch_in, ch_out; K=5, add=true)
x = randn(Float32, (32, 32, ch_in, 1))
y = esp(x)
size(y), typeof(y)
