#!/bin/bash
# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

echo ""
echo "0 -- default"
echo -n "run_id: "
read run_id

# OT parameters
echo ""
echo "0  --  OT"
echo "1  --  UOT"
echo "2  --  POT"
echo -n "choose the OT type: "
read method_choose

case ${method_choose} in
    0 )
        ot_type=ot
        ;;
    1 )
        ot_type=uot
        ;;
    2 )
        ot_type=pot
        ;;
    * )
        echo "The choice of method is illegal!"
        exit 1 
        ;;
esac

SCHEME=ts
METHOD=${SCHEME}${ot_type}
ETA1=0.005
ETA2=1
if [ $ot_type = 'uot' ]
then
    EPSILON=0.01
else
    EPSILON=0
fi
ITER=10000
TEST_INTERVAL=500
M=72
K=2
MASS=0.75
TAU=0.3

for num in 01 02 03 04 05 06 07 08 09 10 11 12
do
    case ${num} in
        01 )
            s_dset_path="./data/office-home/Art.txt"
            t_dset_path="./data/office-home/Clipart.txt"
            output_dir="A2C"
            ;;
        02 )
            s_dset_path="./data/office-home/Art.txt"
            t_dset_path="./data/office-home/Product.txt"
            output_dir="A2P"
            ;;
        03 )
            s_dset_path="./data/office-home/Art.txt"
            t_dset_path="./data/office-home/Real_World.txt"
            output_dir="A2R"
            ;;
        04 )
            s_dset_path="./data/office-home/Clipart.txt"
            t_dset_path="./data/office-home/Art.txt"
            output_dir="C2A"
            ;;
        05 )
            s_dset_path="./data/office-home/Clipart.txt"
            t_dset_path="./data/office-home/Product.txt"
            output_dir="C2P"
            ;;
        06 )
            s_dset_path="./data/office-home/Clipart.txt"
            t_dset_path="./data/office-home/Real_World.txt"
            output_dir="C2R"
            ;;
        07 )
            s_dset_path="./data/office-home/Product.txt"
            t_dset_path="./data/office-home/Art.txt"
            output_dir="P2A"
            ;;
        08 )
            s_dset_path="./data/office-home/Product.txt"
            t_dset_path="./data/office-home/Clipart.txt"
            output_dir="P2C"
            ;;
        09 )
            s_dset_path="./data/office-home/Product.txt"
            t_dset_path="./data/office-home/Real_World.txt"
            output_dir="P2R"
            ;;
        10 )
            s_dset_path="./data/office-home/Real_World.txt"
            t_dset_path="./data/office-home/Art.txt"
            output_dir="R2A"
            ;;
        11 )
            s_dset_path="./data/office-home/Real_World.txt"
            t_dset_path="./data/office-home/Clipart.txt"
            output_dir="R2C"
            ;;
        12 )
            s_dset_path="./data/office-home/Real_World.txt"
            t_dset_path="./data/office-home/Product.txt"
            output_dir="R2P"
            ;;
    esac

    DES="home_${output_dir}_${METHOD}_run${run_id}"
    final_log="home_${METHOD}_run${run_id}"
    output_dir="${DES}_k${K}_m${M}_epsilon${EPSILON}_alpha${ETA1}_lambda${ETA2}_mass${MASS}"

    # train the model
    echo "Begin in ${output_dir}"
    echo "log in ${final_log}_log.txt"
    python train_ts.py --gpu_id ${gpus} \
                    --net ResNet50 \
                    --dset office-home \
                    --test_interval $TEST_INTERVAL \
                    --s_dset_path ${s_dset_path} \
                    --stratify_source \
                    --t_dset_path ${t_dset_path} \
                    --batch_size $M \
                    --output_dir ${output_dir} \
                    --final_log "${final_log}_log.txt" \
                    --stop_step $ITER \
                    --ot_type ${ot_type} \
                    --eta1 $ETA1 \
                    --eta2 $ETA2 \
                    --epsilon $EPSILON \
                    --tau $TAU \
                    --mass $MASS \
                    --k $K
    echo "Finish in ${output_dir}"
done

echo "Training Finished!!!"