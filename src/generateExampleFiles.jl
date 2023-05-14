@info "This script generates the static example train and test files."

using DataFrames, Arrow, CSV, UrlDownload

file_name_train_example = "train_data_example.arrow"
file_name_test_example ="test_data_example.arrow"

print("start download")

    train_datalink_X = "https://tournament.crunchdao.com/data/X_train.csv"
    train_datalink_y = "https://tournament.crunchdao.com/data/y_train.csv"
    test_datalink_X = "https://tournament.crunchdao.com/data/X_test.csv"
    
    train_dataX = urldownload(train_datalink_X)|> DataFrame
    train_dataY =  urldownload(train_datalink_y)|> DataFrame
    test_data  = urldownload(test_datalink_X)|> DataFrame
         
    train_data = innerjoin(train_dataX, train_dataY, on = [:id, :Moons])
    #hcat(train_dataX, train_dataY) previous data
    names(train_data)

    ## multiply by 100
    train_data[!, Cols(x -> contains(x, "Feature"))] = train_data[!, Cols(x -> contains(x, "Feature"))].*100
    test_data[!, Cols(x -> contains(x, "Feature"))] = test_data[!, Cols(x -> contains(x, "Feature"))].*100
   
    ## convert to integer
    train_data[!, Cols(x -> contains(x, "Feature"))] = convert.(Int8, train_data[!, Cols(x -> contains(x, "Feature"))])
    test_data[!, Cols(x -> contains(x, "Feature"))] = convert.(Int8, test_data[!, Cols(x -> contains(x, "Feature"))])
    
    ## write as arrow
    ## train_data |> Arrow.write(joinpath(pwd(),"..\\data", file_name_train)) or alternative below, assuming you call the script from folder "src"
    train_data |> Arrow.write(relpath("..\\data\\"*file_name_train_example))
    ##test_data |> Arrow.write(joinpath(pwd(),"..\\data", file_name_test)) or alternative below, assuming you call the script from folder "src"
    test_data |> Arrow.write(relpath("..\\data\\"*file_name_test_example))