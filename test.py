novelty = 10
fit = -10

frames = 100
min_expl = 0.25
max_expl = 0.75

explore_range = (frames*min_expl, frames*max_expl)

for i in range(frames):

    if i < explore_range[0]:
        fitness = novelty
        print(i, "nov")
    elif  explore_range[0] <= i < explore_range[1]:
        relative_gen = explore_range[1] - explore_range[0]
        gen = i - (frames*min_expl)

        fitness = (gen/relative_gen) * fit + (1-(gen/relative_gen)) * novelty
        print(i, "mix")
        print(gen, relative_gen)
    else:
        fitness = fit
        print(i, "fit")

    print(fitness)

