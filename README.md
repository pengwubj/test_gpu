# test_gpu

test_exp2f_rounding.cu   nvidia exp2f 采用2ulp的精度，这个精度对比对象是采用round to nearest even 的cmath 实现，如果修改rounding mode ，则会超过3ulp， 说明， nv在实现近似拟合的时候，采用了rounding to nearest even 的方式。 
sfu的rounding mode 都是implement defined， 看起来nv的方式和amdgpu一致。
