# Threads.nthreads()
# using SiriusTrack
# model = SiriusTrack.Models.StorageRing.create_accelerator()
# model.cavity_state = true
# model.radiation_state = 1
# model.vchamber_state = true
# p = SiriusTrack.PosModule.Pos(1.0) * 1e-6
# nr_turns = Int(100)
# t = @elapsed begin
#     for i in 1:1:nr_turns
#         for elem in model.lattice
#             SiriusTrack.Tracking.element_pass(elem, p, model)
#         end
#     end
# end
# println(p, "\n", t)
# rnd_part = [[5.572529041751372e-7, 8.116517104935313e-7, 4.72443952657685e-7, 4.6560161114386787e-7, 5.536084278839568e-7, 2.8248805840973556e-9],
#     [8.502407346816891e-7, 2.8275019555659894e-7, 7.591798242340245e-7, 4.7963949743419555e-8, 8.843509773399282e-7, 9.203756007698769e-7],
#     [7.105457828299655e-7, 4.832126502913601e-7, 6.699192875921606e-7, 9.647284382840206e-7, 5.781442336246096e-7, 5.620919269039066e-7],
#     [7.43993398062579e-7, 8.025981446693938e-7, 9.583173564139665e-7, 9.231264983953147e-7, 5.02987882585614e-7, 2.88709177796004e-7],
#     [4.932225377103937e-7, 9.963156252351267e-7, 3.748422758394346e-7, 9.252578858914573e-7, 1.1279250467210233e-8, 2.2388072891554456e-7],
#     [8.95295486787704e-7, 4.061572494052231e-7, 9.204200271508963e-7, 4.7523917219723864e-7, 5.289337272610388e-7, 4.893872536984339e-7],
#     [5.750309379210681e-7, 3.6440762464025386e-7, 2.0127960868718975e-7, 5.218734539883315e-7, 9.920169296667057e-7, 1.8848233468548603e-7],
#     [6.17278820741918e-7, 3.949077050577828e-7, 3.899432663055857e-8, 9.620236347214827e-8, 2.3663539189397407e-7, 5.834459607249801e-7],
#     [4.0535919389329763e-7, 8.736473408520485e-7, 8.15924126704808e-7, 6.297089645179593e-7, 1.2523771537586092e-7, 8.809645951860602e-7],
#     [5.061704060219468e-7, 5.284762354331907e-8, 5.719763962796657e-7, 3.299396035563007e-7, 4.610273974986063e-7, 5.644293502275906e-7],
#     [2.3658721377126877e-7, 1.8876011571998474e-8, 9.69854954816005e-7, 3.5552058222231686e-7, 9.395901255581102e-7, 6.850295643320001e-7],
#     [1.249413052149091e-7, 3.4855719086188675e-7, 2.733911887774397e-7, 9.535987991669914e-8, 4.569115427646537e-7, 8.380818561154815e-7],
#     [1.2377364036972027e-7, 4.4250077943346765e-7, 4.668208377218441e-7, 2.2013919755114153e-7, 2.770461536199458e-7, 9.177961189434769e-8],
#     [4.5808274449792116e-7, 3.78507163578925e-7, 2.887088959753089e-7, 5.189532864290433e-7, 3.849092937884242e-7, 2.7142219822789193e-8],
#     [6.32234156457211e-8, 8.204253983964077e-7, 2.2057736466316557e-7, 2.0720973963581546e-7, 1.446444282628514e-7, 5.05475478651915e-7],
#     [9.562768381885477e-8, 4.2745021931754035e-7, 6.754739687789268e-8, 4.0690817979764134e-7, 3.310522246820401e-7, 4.229501690287127e-7],
#     [2.8517981800087476e-7, 5.38866958239783e-7, 4.751657973710881e-7, 6.887614398206259e-7, 1.132623925625974e-7, 9.631514045495787e-7],
#     [4.659092350948427e-7, 5.825905643180527e-7, 6.910758684946361e-7, 6.683454428330244e-7, 2.9091902020937384e-7, 7.504717220730398e-7],
#     [8.255025272340627e-7, 5.531378765077519e-7, 7.591167298258968e-7, 9.117849048778198e-7, 1.0690506573114267e-7, 1.604263192616986e-7],
#     [4.362256670535025e-7, 8.715484220253027e-7, 9.7748394936691e-7, 6.962119585112838e-7, 9.175688443288585e-7, 9.409914551925419e-7],
#     [5.4193732367260214e-8, 8.586430358292457e-7, 1.2987592733819108e-7, 1.5211288162240166e-7, 6.297173125549794e-7, 8.594163888436016e-7],
#     [5.970779367111178e-7, 9.74933471360585e-8, 5.221388899488811e-7, 9.911575124898328e-7, 2.889011859456645e-7, 6.608259864494224e-7],
#     [2.4651412080990973e-8, 7.608878764858714e-7, 1.7938436328963625e-7, 1.2868657841045017e-7, 7.196349281190938e-8, 5.651918690472758e-7],
#     [5.186954837484856e-7, 8.488922647202763e-7, 7.539803669945343e-7, 9.841877197849424e-8, 8.197574995012287e-7, 5.161964585190754e-7],
#     [2.1276427783853744e-8, 8.288368042289486e-7, 1.0257760380542891e-7, 6.013123491247646e-7, 2.7654751554551215e-7, 6.268670088363238e-8],
#     [2.2044274142844287e-7, 2.5516607551253157e-7, 2.9004500393314333e-7, 1.397721475463637e-7, 8.104477113663655e-7, 9.34731614963975e-7],
#     [6.878823865847262e-7, 3.544060572494706e-7, 7.889061545526866e-7, 6.919598004679606e-8, 6.823791900431415e-7, 5.038465262628005e-7],
#     [4.1486817846338675e-7, 2.8845549333704976e-8, 8.203652812620558e-7, 6.277557913276233e-7, 4.564187553117993e-7, 4.6274927334182254e-8],
#     [3.5262979483809706e-7, 2.9931825628594433e-7, 2.146351611355809e-7, 3.54468581902698e-7, 4.104431835706118e-7, 1.3680312794604376e-7],
#     [5.901660037043712e-7, 4.967177380221477e-7, 7.629798367117923e-7, 5.202481150795135e-7, 2.2391084180199782e-7, 8.880667809811672e-7],
#     [6.209644835965866e-7, 3.608763628893734e-7, 7.312837470649318e-7, 8.28298434442506e-7, 6.102169937132708e-7, 7.52980644151103e-7],
#     [1.3711963460220487e-7, 5.678347000116915e-7, 2.2609435023924627e-7, 3.2893510446854036e-7, 9.403936004269495e-7, 7.317697483789406e-7],
#     [3.311176245096627e-7, 8.65384313364278e-8, 5.397341418988662e-7, 8.270310625104283e-7, 5.144399390948456e-7, 1.8205499988847705e-7],
#     [6.179811251885432e-7, 1.6217278987672477e-7, 1.3763556633694927e-7, 9.791257743049424e-7, 7.160134292363685e-7, 4.02763996840188e-7],
#     [2.984456204580358e-7, 4.859870115130167e-7, 2.2450662086354677e-7, 8.520012824992825e-7, 1.175664610358571e-7, 3.7130661686096634e-7],
#     [7.955011612339788e-7, 9.34312323129948e-7, 3.662664792855594e-7, 8.827254372484716e-7, 8.551917967617385e-7, 8.236434968116522e-7],
#     [9.213825301385386e-7, 6.634605103843849e-7, 8.394655744388551e-7, 9.564084517994806e-7, 4.100591632624788e-7, 7.558841979060285e-7],
#     [6.576058564322924e-7, 1.5436935341009162e-7, 5.992059300064402e-7, 9.681712481416004e-7, 4.126242508613956e-7, 5.470337984977065e-7],
#     [6.766277030386677e-7, 4.3906249132886366e-7, 3.33477110738384e-7, 1.7612416999915892e-8, 8.73189428224694e-7, 7.931335302754402e-7],
#     [2.3118328959281753e-7, 4.408887555074501e-7, 3.363081439961719e-7, 6.131998614696523e-7, 7.708151330456565e-7, 6.731763533265227e-7],
#     [9.405356493395581e-7, 3.056690061745658e-7, 2.6346160174505827e-7, 2.86257363264338e-7, 8.494942687078997e-7, 2.469500443459692e-7],
#     [2.0562551227463166e-7, 6.00014476836328e-8, 9.817828805458005e-7, 7.007782110075015e-7, 5.231580323972917e-7, 3.5273620789836746e-7],
#     [6.4373109337968e-7, 2.983604940194329e-8, 3.236244658147068e-7, 2.777048811992095e-7, 7.323540608420283e-7, 8.85496827857265e-7],
#     [9.686704368067661e-7, 5.637486823196125e-7, 8.564941760279653e-8, 7.907568485031721e-7, 9.015723185778117e-7, 6.573683464802568e-7],
#     [3.4722988908956986e-7, 1.656999358551322e-7, 1.3167165548684866e-7, 8.721643689739605e-7, 6.713952858614444e-7, 4.014547694435563e-7],
#     [8.648701961573144e-7, 2.3926854117137054e-7, 8.009374632222833e-7, 3.343546359793664e-7, 8.082700866581422e-7, 6.694755394757733e-7],
#     [2.74800428569235e-7, 1.0725880331965676e-7, 3.2904201250550135e-7, 1.741586036679188e-8, 5.504247233594916e-7, 7.02778079788298e-7],
#     [6.360971829105247e-7, 9.483074875692901e-7, 3.1431636511993973e-7, 7.612695561693076e-7, 6.539404122664747e-7, 7.345996087187606e-7],
#     [6.58355875599592e-7, 6.90775751435067e-7, 1.826126922393686e-7, 7.620273600318106e-7, 7.624842615777958e-7, 4.378211768755543e-7],
#     [2.969217193866207e-7, 1.0925571287637325e-7, 9.265320087457871e-7, 6.461942090389313e-7, 7.796608991465049e-7, 8.628792755886022e-7]]
    
# model = SiriusTrack.Models.StorageRing.create_accelerator()
# model.cavity_state = true
# model.radiation_state = 1
# model.vchamber_state = true
# p = SiriusTrack.PosModule.Pos(1.0) * 1e-6
# nr_turns = Int(1000)
# timing = [] 
# for i in 1:1:nr_turns
#     t = @elapsed begin
#         SiriusTrack.Tracking.line_pass(model, p, "closed")
#     end
#     push!(timing, t)
# end
# println(p, "\n", sum(timing))
# # println("[")
# # for k in 1:1:50
# #     println(rand(6)*1e-6, ",")
# # end
# # println("]")
# model = SiriusTrack.Models.StorageRing.create_accelerator()
# model.cavity_state = true
# model.radiation_state = 1
# model.vchamber_state = true
# alltt = []
# opart = fill(SiriusTrack.PosModule.Pos(0.0), length(rnd_part))
# for (i, r) in enumerate(rnd_part)
#     p = SiriusTrack.PosModule.Pos(r)
#     nr_turns = Int(100)
#     for j in 1:1:nr_turns
#         SiriusTrack.Tracking.line_pass(model, p, "closed")
#     end
#     opart[i] = copy(p)
# end
# model = SiriusTrack.Models.StorageRing.create_accelerator()
# model.cavity_state = true
# model.radiation_state = 1
# model.vchamber_state = true
# parts = fill(SiriusTrack.PosModule.Pos(0.0), length(rnd_part))
# Threads.@threads for i in 1:1:length(rnd_part)
#     p = SiriusTrack.PosModule.Pos(rnd_part[i])
#     nr_turns = Int(100)
#     for j in 1:1:nr_turns
#         SiriusTrack.Tracking.line_pass(model, p, "closed")
#     end
#     parts[i] = copy(p)
# end
# xl = [s.rx for s in opart];
# xp = [s.rx for s in parts];
# using Plots
# plot()
# plot!(xl, shape=:circle, markersize=2)
# plot!(xp)

using CUDA

using SiriusTrack.PosModule: Pos

struct gpuPos
    rx::Float64
    px::Float64
    ry::Float64
    py::Float64
    de::Float64
    dl::Float64

    function gpuPos(v::Vector{T}) where T
        return new(Float64(v[1]), Float64(v[2]), Float64(v[3]), Float64(v[4]), Float64(v[5]), Float64(v[6]))
    end
end

v = [1, 2, 3, 4, 5, 6]
a = Pos(v)
b = gpuPos(v)
c = CuArray{Float64}(v)

t1 = @elapsed begin
    a.rx = Float64(pi)
end

t2 = @elapsed begin
    CUDA.@allowscalar c[1] = Float64(pi)
end

t3 = @elapsed begin
    @allowscalar() b[1] = Float64(pi)
end