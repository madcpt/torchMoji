datasets=("SS-Youtube" "SS-Twitter")
nb_classes=(2 2)
methods=("new" "full"  "last" "chain-chaw")

for i in ${!datasets[@]}
do
    for method in ${methods[*]}
    do
        python3 examples/finetune.py -m=${method} --dataset=${datasets[$i]} -n=${nb_classes[$i]}
    done
done
