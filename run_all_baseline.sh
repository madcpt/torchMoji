datasets=("SS-Youtube" "SS-Twitter")
nb_classes=(2 2)
methods=("new" "full"  "last" "chain-chaw")

for i in ${!datasets[@]}
do
    for method in ${methods[*]}
    do
        echo -m=${method} --dataset=${datasets[$i]} -n=${nb_classes[$i]}
    done
done
