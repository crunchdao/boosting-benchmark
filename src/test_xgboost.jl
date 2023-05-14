using XGBoost

T = Float64; 
nobs = Int(10000)
num_feat = Int(20)

x_train = rand(T, nobs, num_feat)
y_train = rand(T, size(x_train, 1))

params_xgb = Dict(
    :max_depth => Int(2),
    :eta => 0.01,
    :objective => "reg:squarederror",
    :print_every_n => Int(0)
)

dtrain = DMatrix(x_train, y_train .- 1)
@time m_xgb = xgboost(dtrain, num_round=50,  verbosity=0, param = params_xgb, watchlist = (;));
pred_xgb = XGBoost.predict(m_xgb, x_train);
size(pred_xgb)
