#!/bin/bash


DSET=OfficeHome

for S in {0..3}
do
    for T in {0..3}
    do
        if [ $S != $T ]; then

echo $S, $T, $DSET, warmpot.py
python warmpot.py --dset $DSET --s $S --t $T \
                            --batch_size=65 \
                            --eta1=0.5 \
                            --eta2=7.0 \
                            --eta3=0.25 \
                            --epsilon=7.0 \
                            --mass=0.8 \
                            --beta=0.35 \
                            --max_iterations=5000 \
                            --test_interval=100 \
                            --seed=2020 \
                            --mass_increase_i=2500 \
                            --net="ResNet50"

        fi
    done
done



DSET=ImageNetCaltech
S=0
T=1
echo $S, $T, $DSET, warmpot.py
python warmpot.py --dset $DSET --s $S --t $T \
                            --batch_size=100 \
                            --eta1=0.9182358622250472 \
                            --eta2=5.473751621713613 \
                            --eta3=1 \
                            --epsilon=5.5869302431595145 \
                            --mass=0.07594594824306176 \
                            --beta=0.7158393980865082 \
                            --max_iterations=48000 \
                            --test_interval=100 \
                            --entropy=0.1 \
                            --seed=2020 \
                            --mass_increase_i=2500 \
                            --net="ResNet50"
