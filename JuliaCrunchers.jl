using DataFrames, Arrow, CSV, UrlDownload, Plots
using Statistics, Distributions, StatsBase, EvoTrees
using XGBoost

LIGHTGBM_SOURCE = abspath(pwd()*"/LightGBM-3.2.0")
using LightGBM

Random.seed!(1234);

file_name_train = "train_data.arrow"
file_name_test ="test_data.arrow"
is_download = true
is_test = false
val_frac = 0.3 ## fraction for validation set
T = Float64; ## setting a type for all data calculations; features are saved as integers and converted for the algorithms
T_evo = Float32; ## type for evotress tress. Reduced precision should work well.

@info "is_download set to $(is_download)"

if is_test
    @info "data is reduced as is_test is set to $(is_test)."
end

function read_data(path_in)
    df = open(path_in) do io
       Arrow.Table(io) |> DataFrame
    end;
    return(copy(df)) ## avoids memory mapping for the usual data types https://bkamins.github.io/julialang/2020/11/06/arrow.html
end 

function bin_by_percentiles(x, p)
    @assert issorted(p)
    q = quantile(x, p; sorted = true)
    searchsortedfirst.(Ref(q), x)
end

function bin_by_breaks(x, breaks_in)
    searchsortedfirst.(Ref(breaks_in), x)
end

function get_cor_measure(x,y)
    res = corspearman(x,y)
    println("cor: ", res)
    return(res) ## was cor
end

function rescale(x)
    res = (x.-minimum(x))/(maximum(x)-minimum(x))
end

function add_ranks(x,y)
    res = ordinalrank(x) .+ ordinalrank(y) .- 1
    res = res/(size(x,1) + size(y,1))
    res = quantile.(Normal(), res)
    return( rescale(res)) 
end

function generate_light_gbm_preds(current_target::AbstractString)
    @time LightGBM.fit!(lightGBM_model, Matrix(convert.(T, train_data[train_indices, Cols(x -> startswith(x, "Feature"))])), train_data[train_indices, current_target]; verbosity = -1) 
    ## predict train
    lightgbm_pred_train = LightGBM.predict(lightGBM_model, Matrix(convert.(T, train_data[train_indices, Cols(x -> startswith(x, "Feature"))])));
    train_cor = get_cor_measure(lightgbm_pred_train, train_data[train_indices, current_target]);
    println("Train performance ", current_target, " : ", train_cor[1,1])

    ## validation
    lightgbm_pred_val = LightGBM.predict(lightGBM_model, Matrix(convert.(T, train_data[val_indices, Cols(x -> startswith(x, "Feature"))])));
    val_cor = get_cor_measure(lightgbm_pred_val, train_data[val_indices, current_target]);
    println("Val performance ", current_target, " : ", val_cor[1,1])

    ## predict test
    lightgbm_pred_live = LightGBM.predict(lightGBM_model, Matrix(convert.(T, test_data[:, Cols(x -> startswith(x, "Feature"))])));

    return lightgbm_pred_train[:,1], lightgbm_pred_val[:,1], lightgbm_pred_live[:,1]
end 

function generate_xgboost_preds(current_target::AbstractString)
    @time xgboost_model = xgboost( DMatrix(convert.(T, train_data[train_indices, Cols(x -> startswith(x, "Feature"))]), train_data[train_indices, current_target]), 
                    num_round = get!(xgboost_pars, "num_iterations", 100),
                    max_depth = get!(xgboost_pars, "max_depth", 5),
                    subsample = get!(xgboost_pars, "subsample", 1), 
                    colsample_bytree = get!(xgboost_pars, "colsample_bytree", 1),
                    tree_method = get!(xgboost_pars, "tree_method", "hist"), 
                    max_bin = get!(xgboost_pars, "max_bin", 64), 
                    objective = "reg:squarederror", eval_metric = "rmse", print_every_n = 100, verbosity = 0, verbose_eval = 0, nthread=nthread, watchlist = (;))
    
    # train, rescaled to be in [0,1]
    xgboost_pred_train = XGBoost.predict(xgboost_model, DMatrix(convert.(T, train_data[train_indices, Cols(x -> startswith(x, "Feature"))]))) |> rescale;
   
    train_cor = get_cor_measure(xgboost_pred_train, train_data[train_indices, current_target]);
    println("Train performance ", current_target, " : ", train_cor[1,1])

    # validation
    xgboost_pred_val = XGBoost.predict(xgboost_model, DMatrix(convert.(T, train_data[val_indices, Cols(x -> startswith(x, "Feature"))]))) |> rescale;
    val_cor = get_cor_measure(xgboost_pred_val, train_data[val_indices, current_target]);
    println("Validation performance ", current_target, " : ", val_cor[1,1])

    # test        
    xgboost_pred_live = XGBoost.predict(xgboost_model, DMatrix(convert.(T, test_data[:, Cols(x -> startswith(x, "Feature"))]))) |> rescale;

    return xgboost_pred_train, xgboost_pred_val, xgboost_pred_live, xgboost_model
end 


function generate_evotree_preds(current_target::AbstractString)
    @time m_evo = fit_evotree(params_evo; x_train = Matrix(convert.(T, train_data[train_indices, Cols(x -> startswith(x, "Feature"))])), y_train = train_data[train_indices, current_target], metric=:mae, print_every_n=500, 
                             x_eval = Matrix(convert.(T, train_data[train_indices, Cols(x -> startswith(x, "Feature"))])), y_eval = train_data[train_indices, current_target]);

    evotree_pred_train = EvoTrees.predict(m_evo, Matrix(convert.(T, train_data[train_indices, Cols(x -> startswith(x, "Feature"))])));
    train_cor = get_cor_measure(evotree_pred_train, train_data[train_indices, current_target]);
    println("Train performance ", current_target, " : ", train_cor[1,1])

    ## validation
    evotree_pred_val = EvoTrees.predict(m_evo, Matrix(convert.(T, train_data[val_indices, Cols(x -> startswith(x, "Feature"))])));
    val_cor = get_cor_measure(evotree_pred_val, train_data[val_indices, current_target]);
    println("Validation performance ", current_target, " : ", val_cor[1,1])

    ## predict test
    evotree_pred_live = EvoTrees.predict(m_evo, Matrix(convert.(T, test_data[:, Cols(x -> startswith(x, "Feature"))])));

    return evotree_pred_train, evotree_pred_val, evotree_pred_live, m_evo
end 

function append_to_res_table(current_model::AbstractString, current_target::AbstractString, valcor::AbstractFloat)
    ## appends results to the results table
    append!(res_table, DataFrame(model = current_model, target = current_target, valcor = valcor))
end

if is_download 
    print("start download")
    ## old format
    #train_datalink_X = "https://tournament.datacrunch.com/data/X_train.csv"  
    #train_datalink_y = "https://tournament.datacrunch.com/data/y_train.csv"
    #hackathon_data_link = "https://tournament.datacrunch.com/data/X_test.csv"
    ## new single data set. Now it is csv, later use parquet format
    train_datalink_X = "https://tournament.crunchdao.com/data/X_train_one_dataset.csv"
    train_datalink_y = "https://tournament.crunchdao.com/data/y_train_one_dataset.csv"
    test_datalink_X = "https://tournament.crunchdao.com/data/X_test_one_dataset.csv"
    
    train_dataX = urldownload(train_datalink_X)|> DataFrame
    train_dataY =  urldownload(train_datalink_y)|> DataFrame
    test_data  = urldownload(test_datalink_X)|> DataFrame
         
    train_data = innerjoin(train_dataX, train_dataY, on = [:id, :Moons])
    #hcat(train_dataX, train_dataY) previous data
    names(train_data)

    ## multiply by 100
    train_data[!, Cols(x -> startswith(x, "Feature"))] = train_data[!, Cols(x -> startswith(x, "Feature"))].*100
    test_data[!, Cols(x -> startswith(x, "Feature"))] = test_data[!, Cols(x -> startswith(x, "Feature"))].*100
    ## convert to integer
    train_data[!, Cols(x -> startswith(x, "Feature"))] = convert.(Int8, train_data[!, Cols(x -> startswith(x, "Feature"))])
    test_data[!, Cols(x -> startswith(x, "Feature"))] = convert.(Int8, test_data[!, Cols(x -> startswith(x, "Feature"))])
    
    ## write as arrow
    train_data |> Arrow.write(joinpath(pwd(),"data", file_name_train))
    test_data |> Arrow.write(joinpath(pwd(),"data", file_name_test))
    print("\n data is downloaded")
else  
    train_data = read_data(joinpath(pwd(),"data", file_name_train))
    test_data = read_data(joinpath(pwd(),"data", file_name_test))
end

if is_test ## reduce size by factor 10 for tests
    train_data = train_data[1:10:size(train_data,1),:]
end

names(train_data)
names(test_data)

## using a symbol to select a column, just illustration
select(train_data,:Moons) |> unique
## using regex to select columns, just illustration
select(train_data, r"Feature")
max_moon = maximum(eachrow(select(train_data,:Moons)))[1];

@info "Train data with: $max_moon moons."
val_moons =  sample(0:max_moon, floor(Int, val_frac*max_moon), replace = false)
max_m, min_m = maximum(val_moons), minimum(val_moons)

@info "Validation moons are from moon $min_m to moon $max_m ."

val_indices = [i for i in 1:size(train_data, 1) if train_data[i,:Moons] in val_moons]
train_indices = setdiff(1:size(train_data, 1), val_indices)

train_data[train_indices,:]
train_data[val_indices,:]

@info "Val data with share of $(val_frac*100)%."

## initialze the results table
res_table = DataFrame(model = String[], target = String[], valcor = Float64[])

@info "Training starts here. Timing uses the built-in timing makro and includes initial compilation. This should not harm as we are interest in overall runtime."

## starting lightgbm

@info "training lightgbm" 
## https://github.com/IQVIA-ML/LightGBM.jl

# Create an estimator with the desired parameters—leave other parameters at the default values.

lightgbm_pars = Dict([
    ("num_iterations", 2000), 
    ("learning_rate", .01), 
    ("max_depth", 5),
    ("feature_fraction", 0.5),
    ("bagging_fraction", 1.0)
    ])


lightGBM_model = LGBMRegression(
    objective = "regression",
    num_iterations = get!(lightgbm_pars, "num_iterations", 0),
    learning_rate = get!(lightgbm_pars, "learning_rate", 0),
    max_depth = get!(lightgbm_pars, "max_depth", 0),
    feature_fraction = get!(lightgbm_pars, "feature_fraction", 0),
    bagging_fraction = get!(lightgbm_pars, "bagging_fraction", 0),
    metric = ["l2"]
)

## target_w
current_target = "target_w"
lightgbm_pred_train_w, lightgbm_pred_val_w, lightgbm_pred_live_w = generate_light_gbm_preds(current_target);
append_to_res_table("lightgbm",current_target, get_cor_measure(lightgbm_pred_val_w, train_data[val_indices, current_target]))


## target_r
current_target = "target_r"
lightgbm_pred_train_r, lightgbm_pred_val_r, lightgbm_pred_live_r = generate_light_gbm_preds(current_target);
append_to_res_table("lightgbm",current_target, get_cor_measure(lightgbm_pred_val_r, train_data[val_indices, current_target]))


## target_g
current_target = "target_g"
lightgbm_pred_train_g, lightgbm_pred_val_g, lightgbm_pred_live_g = generate_light_gbm_preds(current_target);
append_to_res_table("lightgbm",current_target, get_cor_measure(lightgbm_pred_val_g, train_data[val_indices, current_target]))


## target_b
current_target = "target_b"
lightgbm_pred_train_b, lightgbm_pred_val_b, lightgbm_pred_live_b = generate_light_gbm_preds(current_target);
append_to_res_table("lightgbm",current_target, get_cor_measure(lightgbm_pred_val_b, train_data[val_indices, current_target]))

## some importance analysis for illustration
LightGBM.gain_importance(lightGBM_model)

## train xgboost

@info "training xgboost" 
## https://dmlc.github.io/XGBoost.jl/dev/api/#Data-Input
## https://morioh.com/p/268c31c0e328

nthread = Base.Threads.nthreads()
println("Training xgboost with $(nthread) thread(s).")
xgboost_pars = Dict([
    ("num_iterations", 2000), 
    ("learning_rate", .01), 
    ("max_depth", 5),
    ("subsample", 1.0),
    ("colsample_bytree", .5),
    ("tree_method", "hist"),
    ("max_bin", 64)
    ])


## target_w
current_target = "target_w"
xgboost_pred_train_w, xgboost_pred_val_w, xgboost_pred_live_w, xgboost_model = generate_xgboost_preds(current_target);
append_to_res_table("xgboost", current_target, get_cor_measure(xgboost_pred_val_w, train_data[val_indices, current_target]))

## target_r
current_target = "target_r"
xgboost_pred_train_r, xgboost_pred_val_r, xgboost_pred_live_r, xgboost_model = generate_xgboost_preds(current_target);
append_to_res_table("xgboost", current_target, get_cor_measure(xgboost_pred_val_r, train_data[val_indices, current_target]))

## target_g
current_target = "target_g"
xgboost_pred_train_g, xgboost_pred_val_g, xgboost_pred_live_g, xgboost_model = generate_xgboost_preds(current_target);
append_to_res_table("xgboost", current_target, get_cor_measure(xgboost_pred_val_g, train_data[val_indices, current_target]))

## target_b
current_target = "target_b"
xgboost_pred_train_b, xgboost_pred_val_b, xgboost_pred_live_b, xgboost_model = generate_xgboost_preds(current_target);
append_to_res_table("xgboost", current_target, get_cor_measure(xgboost_pred_val_b, train_data[val_indices, current_target]))

imp = DataFrame(importancetable(xgboost_model))
importancereport(xgboost_model)
sort!(imp, [:total_gain, :total_cover], rev = false)
first(imp,4)

## train evotrees

@info "training evotrees" 
nrounds = Int(2000)
params_evo = EvoTreeRegressor(
    T=T_evo,
    loss=:linear,
    nrounds=nrounds,
    alpha=0.5,
    lambda=0.0,
    gamma=0.0,
    eta=0.01,
    max_depth=6, ## 1 more than xgboost
    min_weight=1.0,
    rowsample=1.0,
    colsample=0.5,
    nbins=64, 
    rng = 123,
    device = "cpu"
    )


## target_w
current_target = "target_w"
evotree_pred_train_w, evotree_pred_val_w, evotree_pred_live_w, m_evo = generate_evotree_preds(current_target);
append_to_res_table("evotrees", current_target, get_cor_measure(evotree_pred_val_w, train_data[val_indices, current_target]))

## target_r
current_target = "target_r"
evotree_pred_train_r, evotree_pred_val_r, evotree_pred_live_r, m_evo = generate_evotree_preds(current_target);
append_to_res_table("evotrees", current_target, get_cor_measure(evotree_pred_val_r, train_data[val_indices, current_target]))

## target_g
current_target = "target_g"
evotree_pred_train_g, evotree_pred_val_g, evotree_pred_live_g, m_evo = generate_evotree_preds(current_target);
append_to_res_table("evotrees", current_target, get_cor_measure(evotree_pred_val_g, train_data[val_indices, current_target]))

## target_b
current_target = "target_b"
evotree_pred_train_b, evotree_pred_val_b, evotree_pred_live_b, m_evo = generate_evotree_preds(current_target);
append_to_res_table("evotrees", current_target, get_cor_measure(evotree_pred_val_b, train_data[val_indices, current_target]))

## trying random forrest
## https://stackoverflow.com/questions/73028061/randomforestregressor-in-julia


## tests
@info "Plots live predictions:"

p1 = histogram(lightgbm_pred_live_r, title="lightgbm")
p2 = histogram(evotree_pred_live_r, title="evotree")
p3 = histogram(xgboost_pred_live_r, title="xgboost")
plot(p1, p2, p3, layout=(1, 3), legend=false)


@info "Correlatons between various live predictions:"

@info "lighgbm and evotree:"
cor(lightgbm_pred_live_r, evotree_pred_live_r) |> println
cor(lightgbm_pred_live_g, evotree_pred_live_g) |> println
cor(lightgbm_pred_live_b, evotree_pred_live_b) |> println

@info "lighgbm and xgboost:"
cor(lightgbm_pred_live_r, xgboost_pred_live_r) |> println 
cor(lightgbm_pred_live_g, xgboost_pred_live_g) |> println
cor(lightgbm_pred_live_b, xgboost_pred_live_b) |> println

@info "evotree and xgboost:"

cor(evotree_pred_live_r, xgboost_pred_live_r) |> println
cor(evotree_pred_live_g, xgboost_pred_live_g) |> println
cor(evotree_pred_live_b, xgboost_pred_live_b) |> println

@info "evotree is strongly correlated with lightgbm (cor > 0.9) but much less with with xgboost (cor 0.3)"

## average predictions based on ranks

@info "assessment of increase in performance on the validation data set:"

## w

current_target = "target_w"
@info current_target

res = add_ranks(xgboost_pred_val_w, lightgbm_pred_val_w);
@info "xgboost + lightgbm"
get_cor_measure(xgboost_pred_val_w, train_data[val_indices, current_target]);
get_cor_measure(lightgbm_pred_val_w, train_data[val_indices, current_target]);
get_cor_measure(res, train_data[val_indices, current_target]);

append_to_res_table("xgboost + lightgbm", current_target, get_cor_measure(res, train_data[val_indices, current_target]))


res = add_ranks(evotree_pred_val_w, lightgbm_pred_val_w);
@info "evotrees + lightgbm"
get_cor_measure(evotree_pred_val_w, train_data[val_indices, current_target]);
get_cor_measure(lightgbm_pred_val_w, train_data[val_indices, current_target]);
get_cor_measure(res, train_data[val_indices, current_target]);

append_to_res_table("evotrees + lightgbm", current_target, get_cor_measure(res, train_data[val_indices, current_target]))

@info "Only xgboost can handle this target out of sample."

## r

current_target = "target_r"
@info current_target

res = add_ranks(xgboost_pred_val_r, lightgbm_pred_val_r);
get_cor_measure(xgboost_pred_val_r, train_data[val_indices, current_target])
get_cor_measure(lightgbm_pred_val_r, train_data[val_indices, current_target])
get_cor_measure(res, train_data[val_indices, current_target])

append_to_res_table("xgboost + lightgbm", current_target, get_cor_measure(res, train_data[val_indices, current_target]))

res = add_ranks(evotree_pred_val_r, lightgbm_pred_val_r);
get_cor_measure(evotree_pred_val_r, train_data[val_indices, current_target])
get_cor_measure(lightgbm_pred_val_r, train_data[val_indices, current_target])
get_cor_measure(res, train_data[val_indices, current_target])

append_to_res_table("evotrees + lightgbm", current_target, get_cor_measure(res, train_data[val_indices, current_target]))

@info "The performance increased for the ensemble with xgboost due to low correlation and the high performance of xgboost."

## g

current_target = "target_g"
@info current_target

res = add_ranks(xgboost_pred_val_g, lightgbm_pred_val_g);
get_cor_measure(xgboost_pred_val_g, train_data[val_indices, current_target])
get_cor_measure(lightgbm_pred_val_g, train_data[val_indices, current_target])
get_cor_measure(res, train_data[val_indices, current_target])

append_to_res_table("xgboost + lightgbm", current_target, get_cor_measure(res, train_data[val_indices, current_target]))


res = add_ranks(evotree_pred_val_g, lightgbm_pred_val_g);
get_cor_measure(evotree_pred_val_g, train_data[val_indices, current_target])
get_cor_measure(lightgbm_pred_val_g, train_data[val_indices, current_target])
get_cor_measure(res, train_data[val_indices, current_target])

append_to_res_table("evotrees + lightgbm", current_target, get_cor_measure(res, train_data[val_indices, current_target]))



## b
current_target = "target_b"
@info current_target

res = add_ranks(xgboost_pred_val_b, lightgbm_pred_val_b);
get_cor_measure(xgboost_pred_val_b, train_data[val_indices, current_target])
get_cor_measure(lightgbm_pred_val_b, train_data[val_indices, current_target])
get_cor_measure(res, train_data[val_indices, current_target])

append_to_res_table("xgboost + lightgbm", current_target, get_cor_measure(res, train_data[val_indices, current_target]))

res = add_ranks(evotree_pred_val_b, lightgbm_pred_val_b);
get_cor_measure(evotree_pred_val_b, train_data[val_indices, current_target])
get_cor_measure(lightgbm_pred_val_b, train_data[val_indices, current_target])
get_cor_measure(res, train_data[val_indices, current_target])

append_to_res_table("evotrees + lightgbm", current_target, get_cor_measure(res, train_data[val_indices, current_target]))


sort(filter(:target => ==("target_w"), res_table), :valcor, rev=true)
sort(filter(:target => ==("target_r"), res_table), :valcor, rev=true)
sort(filter(:target => ==("target_g"), res_table), :valcor, rev=true)
sort(filter(:target => ==("target_b"), res_table), :valcor, rev=true)


CSV.write("results_table.csv", res_table)
CSV.write("results_table_target_w.csv", sort(filter(:target => ==("target_w"), res_table), :valcor, rev=true))
## checking package versions
## using Pkg
## call ] add EvoTrees#juliaConnectoR
## Pkg.status()


