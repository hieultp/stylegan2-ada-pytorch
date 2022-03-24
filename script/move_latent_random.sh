for i in {0..4}
do
    python boundary_mover.py \
        --network=runs/00000-bedroom_256_train_mixed-mirror-stylegan2-kimg15000/network-snapshot-015000.pkl \
        --classifier=outputs/bedroom_256_classification/2022-03-21-16-38-14/ckpts/vgg19_bn-epoch11-step9999.ckpt \
        --target-cls=1 \
        --outdir=results/boundary_mover/random/without_sim \
        --num-steps=500
done

for dir in results/boundary_mover/random/without_sim/*
do
    seed=${dir/-cls-1/}
    seed=${seed/"results/boundary_mover/random/without_sim/"/}
    python boundary_mover.py \
        --network=runs/00000-bedroom_256_train_mixed-mirror-stylegan2-kimg15000/network-snapshot-015000.pkl \
        --classifier=outputs/bedroom_256_classification/2022-03-21-16-38-14/ckpts/vgg19_bn-epoch11-step9999.ckpt \
        --measurer=outputs/bedroom_256_similarity/2022-03-22-16-31-15/ckpts/regnet_x_8gf-epoch23-step9999.ckpt \
        --target-cls=1 \
        --outdir=results/boundary_mover/random/with_sim \
        --num-steps=500 \
        --seed=$seed
done
