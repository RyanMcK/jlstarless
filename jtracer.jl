@everywhere using Images
@everywhere using ConfParser
@everywhere import PyPlot
@everywhere const plt = PyPlot
@everywhere using PyCall

@everywhere begin
function airy_disk(x)
    return (2.0*besselj1(x) ./ (x)).^2.0
end

meshgrid(v::AbstractVector) = meshgrid(v, v)

function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end

function generate_kernel(scale, sz)
    x = collect(-sz:1.0:sz)
    y = collect(-sz:1.0:sz)

    xs, ys = meshgrid(x, y)
    kernel = zeros(size(xs, 1), size(xs, 2), 3)

    r = sqrt(xs^2.0 + ys^2.0) + 0.000001
    kernel[:, :, :] = airy_disk(reshape(r, size(r, 1), size(r, 2), 1) ./ reshape(scale, 1, 1, size(scale, 1)))

    kernel = kernel ./ sum(kernel, (1, 2))

    return kernel
end

SPECTRUM = [1.0, 0.86, 0.61]

@pyimport scipy.signal as sig
function airy_convolve(arr, radius, kernel_radius=25.0)
    kernel = generate_kernel(radius * SPECTRUM, kernel_radius)

    out = zeros(size(arr, 1), size(arr, 2), 3)
    for i=1:3
        out[:, :, i] = sig.convolve2d(arr[:, :, i], kernel[:, :, i], mode="same", boundary="symm")
    end

    return out
end

METH_LEAPFROG = 0
METH_RK4 = 1

LOFI = 0

DISABLE_DISPLAY = 0
DISABLE_SHUFFLING = 0

NTHREADS = 1

OVERRIDE_RES = 0

SCENE_FNAME = "scenes/jdefault.scene"

CHUNKSIZE = 524

FOGSKIP = 1
METHOD = METH_RK4

ST_NONE = 0
ST_TEXTURE = 1
ST_FINAL = 2

DT_NONE = 0
DT_TEXTURE = 1
DT_SOLID = 2
DT_GRID = 3
DT_BLACKBODY = 4

cfp = ConfParse(SCENE_FNAME)
parse_conf!(cfp)

if OVERRIDE_RES == 0
    RESOLUTION = [parse(Int, x) for x in retrieve(cfp, "lofi", "Resolution")]
end
NITER = parse(Int, retrieve(cfp, "lofi", "Iterations"))
STEP = parse(Float64, retrieve(cfp, "lofi", "Stepsize"))

if LOFI == 0
    if OVERRIDE_RES == 0
        RESOLUTION = [parse(Int, x) for x in retrieve(cfp, "hifi", "Resolution")]
    end
    NITER = parse(Int, retrieve(cfp, "hifi", "Iterations"))
    STEP = parse(Float64, retrieve(cfp, "hifi", "Stepsize"))
end

CAMERA_POS = [parse(Float64, x) for x in retrieve(cfp, "geometry", "Cameraposition")]
TANFOV = parse(Float64, retrieve(cfp, "geometry", "Fieldofview"))
LOOKAT = [parse(Float64, x) for x in retrieve(cfp, "geometry", "Lookat")]
UPVEC = [parse(Float64, x) for x in retrieve(cfp, "geometry", "Upvector")]
DISTORT = parse(Int, retrieve(cfp, "geometry", "Distort"))
DISKINNER = parse(Float64, retrieve(cfp, "geometry", "Diskinner"))
DISKOUTER = parse(Float64, retrieve(cfp, "geometry", "Diskouter"))

DISK_MULTIPLIER = parse(Float64, retrieve(cfp, "materials", "Diskmultiplier"))
DISK_INTENSITY_DO = parse(Int, retrieve(cfp, "materials", "Diskintensitydo"))
REDSHIFT = parse(Float64, retrieve(cfp, "materials", "Redshift"))

GAIN = parse(Float64, retrieve(cfp, "materials", "Gain"))
NORMALIZE = parse(Float64, retrieve(cfp, "materials", "Normalize"))

BLOOMCUT = parse(Float64, retrieve(cfp, "materials", "Bloomcut"))

HORIZON_GRID = parse(Int, retrieve(cfp, "materials", "Horizongrid"))
DISK_TEXTURE = retrieve(cfp, "materials", "Disktexture")
SKY_TEXTURE = retrieve(cfp, "materials", "Skytexture")
SKYDISK_RATIO = parse(Float64, retrieve(cfp, "materials", "Skydiskratio"))
FOGDO = parse(Int, retrieve(cfp, "materials", "Fogdo"))
BLURDO = parse(Int, retrieve(cfp, "materials", "Blurdo"))
AIRY_BLOOM = parse(Int, retrieve(cfp, "materials", "Airy_bloom"))
AIRY_RADIUS = parse(Float64, retrieve(cfp, "materials", "Airy_radius"))
FOGMULT = parse(Float64, retrieve(cfp, "materials", "Fogmult"))

SRGBOUT = parse(Int, retrieve(cfp, "materials", "sRGBOut"))
SRGBIN = parse(Int, retrieve(cfp, "materials", "sRGBIn"))

if SKY_TEXTURE == "none"
    SKY_TEXTURE_INT = ST_NONE
elseif SKY_TEXTURE == "texture"
    SKY_TEXTURE_INT = ST_TEXTURE
else
    SKY_TEXTURE_INT = ST_FINAL
end

if DISK_TEXTURE == "none"
    DISK_TEXTURE_INT = DT_NONE
elseif DISK_TEXTURE == "texture"
    DISK_TEXTURE_INT = DT_TEXTURE
elseif DISK_TEXTURE == "solid"
    DISK_TEXTURE_INT = DT_SOLID
elseif DISK_TEXTURE == "grid"
    DISK_TEXTURE_INT = DT_GRID
else
    DISK_TEXTURE_INT = DT_BLACKBODY
end

if norm(CAMERA_POS) <= 1.0
    println("[ERROR] The observer's 4-velocity is not timelike.")
    exit(1)
end

DISKINNERSQR = DISKINNER*DISKINNER
DISKOUTERSQR = DISKOUTER*DISKOUTER

if isdir("tests") == false
    mkdir("tests")
end

function rgbtosrgb(arr)
    println("RGB -> sRGB")
    mask = arr .> 0.0031308
    arr[mask] .^= (1.0/2.4)
    arr[mask] *= 1.055
    arr[mask] -= 0.055
    arr[~mask] *= 12.92
end

function srgbtorgb(arr)
    println("sRGB -> RGB")
    mask = arr .> 0.04045
    arr[mask] += 0.055
    arr[mask] ./= 1.055
    arr[mask] .^= 2.4
    arr[~mask] ./= 12.92
end

@pyimport scipy.misc as spm
if SKY_TEXTURE == "texture"
    texarr_sky = separate(imread("textures/bgedit.jpg")) / 1.0
    texarr_sky = texarr_sky[:, :, 1:3] / 1.0
    if SRGBIN != 0
        srgbtorgb(texarr_sky)
    end
    if LOFI == 0
        texarr_sky = spm.imresize(texarr_sky, 2.0, interp="bicubic") / 255.0
    end
end

if DISK_TEXTURE == "texture"
    texarr_disk = separate(imread("textures/adisk.jpg"))/1.0
    texarr_disk = texarr_disk[:, :, 1:3] / 1.0
elseif DISK_TEXTURE == "test"
    texarr_disk = separate(imread("textures/adisktest.jpg"))/1.0
    texarr_disk = texarr_disk[:, :, 1:3] / 1.0
end
if SRGBIN != 0
    srgbtorgb(texarr_disk)
end

function lookup(texarr, uvarrin)
    uvarr = clamp(uvarrin, 0.0, 0.999)
    uvarr[:, 1] *= Float64(size(texarr, 2))
    uvarr[:, 2] *= Float64(size(texarr, 1))

    uvarr = trunc(Int, uvarr) + 1

    out = zeros(size(uvarr, 1), 3)
    for tmpi=1:size(uvarr, 1)
        out[tmpi, :] = texarr[uvarr[tmpi, 2], uvarr[tmpi, 1], :]
    end
    return out
end

FRONTVEC = (LOOKAT-CAMERA_POS)
FRONTVEC = FRONTVEC / norm(FRONTVEC)

LEFTVEC = cross(UPVEC,FRONTVEC)
LEFTVEC = LEFTVEC/norm(LEFTVEC)

NUPVEC = cross(FRONTVEC,LEFTVEC)

viewMatrix = zeros(3,3)

viewMatrix[:,1] = LEFTVEC
viewMatrix[:,2] = NUPVEC
viewMatrix[:,3] = FRONTVEC

pixelindices = collect(1:RESOLUTION[1]*RESOLUTION[2])

numPixels = size(pixelindices, 1)

onesvec = ones(numPixels)
ones3vec = ones(numPixels, 3)
tmparr = [0.0, 1.0, 0.0]
UPFIELD = onesvec * tmparr'

ransample = rand(numPixels)

function vnorm(vec)
    tmpn = length(size(vec))
    if tmpn == 2
        return [norm(sub(vec,i,:)) for i in 1:size(vec,1)]
    else
        res = zeros(size(vec, 1), size(vec, 2))

        for tmpi=1:size(vec, 1)
            for tmpj=1:size(vec, 2)
                res[tmpi, tmpj] = norm(reshape(vec[tmpj, :, tmpi], 3))
            end
        end
        return res
    end
end

function normalize(vec)
    tmp = vnorm(vec)
    return vec ./ reshape(tmp, size(tmp, 1), 1)
end

function sqrnorm(vec)
    out = zeros(size(vec, 1))
    sumabs2!(out, vec)
    return out
end

function fivenorm(vec)
    out = zeros(size(vec, 1))
    sumabs2!(out, vec)
    out .*= sqrt(sqrt(out))
    out .*= out
    return out
end

function RK4f!(k, y, h2)
    k[:, 1:3] = y[:, 4:6]

    tmp = zeros(size(y, 1))
    sumabs2!(tmp, y[:, 1:3])
    tmp .*= sqrt(sqrt(tmp))
    tmp .*= tmp

    k[:, 4:6] = y[:, 1:3]
    k[:, 4:6] ./= tmp
    k[:, 4:6] .*= h2
    k[:, 4:6] .*= -1.5
    return
end

function blendcolors(cb, balpha, ca, aalpha)
    tmp = (balpha .* (1.0 - aalpha))
    return ca + cb .* reshape(tmp, size(tmp, 1), 1)
end

function blendalpha(balpha, aalpha)
    return aalpha + balpha .* (1.0 - aalpha)
end

function saveToImg(arr, fname)
    println("Saving image")
    imgout = arr
    imgout = clamp(imgout, 0.0, 1.0)
    if SRGBOUT != 0
        rgbtosrgb(imgout)
    end
    imgout = reshape(imgout, RESOLUTION[1], RESOLUTION[2], 3)
    imwrite(convert(Image, imgout), fname)
end

function saveToImgBool(arr, fname)
    tmp = [1.0, 1.0, 1.0]
    saveToImg(arr * tmp', fname)
end

end
tic()

if DISABLE_SHUFFLING == 0
    shuffle!(pixelindices)
end
n = div(numPixels, CHUNKSIZE)
chunksM = reshape(pixelindices, div(length(pixelindices), n), n)'

chunks = Vector{Int}[]
for j=1:size(chunksM, 1)
    push!(chunks, reshape(chunksM[j, :], CHUNKSIZE))
end

NCHUNKS = size(chunks, 1)

println("Using ", NCHUNKS, " chunks of size ", CHUNKSIZE)

total_colour_buffer_preproc_shared = SharedArray(Float64, (numPixels, 3))
total_colour_buffer_preproc = total_colour_buffer_preproc_shared

shuffle!(chunks)

schedules = Vector{Vector{Int}}[]

q = div(NCHUNKS, NTHREADS)
r = mod(NCHUNKS, NTHREADS)

indices = [q*i + min(i, r) + 1 for i=0:NTHREADS]

for i=1:NTHREADS
    push!(schedules, chunks[indices[i]:indices[i+1]-1])
    println("Thread ", i, " is given ", size(schedules[i], 1), " chunks")
end

@everywhere function raytrace_schedule(i, schedule, total_shared)
    if size(schedule, 1) == 0
        return
    end

    total_colour_buffer_preproc = total_shared

    cn = 0
    for chunk in schedule
        cn += 1
        numChunk = size(chunk, 1)
        tmp_ones = ones(numChunk)
        tmp_ones3 = ones(numChunk, 3)
        tmp_y = [0.0, 1.0, 0.0]
        tmp_zero = zeros(3)
        UPFIELD = tmp_ones * tmp_y'
        BLACK = tmp_ones * tmp_zero'

        x = chunk % RESOLUTION[1]
        y = chunk / RESOLUTION[1]

        view = zeros(numChunk, 3)
        view[:, 1] = map(Float64, x) / RESOLUTION[1] - 0.5
        view[:, 2] = ((-1.0*map(Float64, y) / RESOLUTION[2] + 0.5) * RESOLUTION[2]) / RESOLUTION[1]
        view[:, 3] = 1.0

        view[:, 1] *= TANFOV
        view[:, 2] *= TANFOV

        view = (viewMatrix * view')'

        point = tmp_ones * CAMERA_POS'

        normview = normalize(view)

        velocity = copy(normview)

        object_colour = zeros(numChunk, 3)
        object_alpha = zeros(numChunk)

        tc = zeros(point)
        for tci=1:size(point, 1)
            tc[tci, :] = cross(reshape(point[tci, :], 3), reshape(velocity[tci, :], 3))
        end
        h2 = sqrnorm(tc)
        h2 = reshape(h2, size(h2, 1), 1)

        pointsqr = copy(tmp_ones3)

        for it=0:NITER-1
            oldpoint = copy(point)

            if METHOD == METH_LEAPFROG
                point += velocity * STEP
                if DISTORT != 0
                    tmp = (sqrnorm(point))^2.5
                    accel = -1.5 * h2 * point / reshape(tmp, size(tmp, 1), 1)
                    velocity += accel * STEP
                end
            elseif METHOD == METH_RK4
                if DISTORT != 0
                    rkstep = STEP

                    y = zeros(numChunk, 6)
                    y[:, 1:3] = point
                    y[:, 4:6] = velocity
                    k1 = zeros(numChunk, 6)
                    k2 = zeros(numChunk, 6)
                    k3 = zeros(numChunk, 6)
                    k4 = zeros(numChunk, 6)

                    y1 = copy(y)
                    k1[:, 1:3] = y1[:, 4:6]
                    tmp = zeros(size(y1, 1))
                    sumabs2!(tmp, y1[:, 1:3])
                    tmp .*= sqrt(sqrt(tmp))
                    tmp .*= tmp
                    k1[:, 4:6] = y1[:, 1:3]
                    k1[:, 4:6] ./= tmp
                    k1[:, 4:6] .*= h2
                    k1[:, 4:6] .*= -1.5

                    y2 = copy(y)
                    y2 += (0.5*rkstep*k1)
                    k2[:, 1:3] = y2[:, 4:6]
                    tmp = zeros(size(y2, 1))
                    sumabs2!(tmp, y2[:, 1:3])
                    tmp .*= sqrt(sqrt(tmp))
                    tmp .*= tmp
                    k2[:, 4:6] = y2[:, 1:3]
                    k2[:, 4:6] ./= tmp
                    k2[:, 4:6] .*= h2
                    k2[:, 4:6] .*= -1.5

                    y3 = copy(y)
                    y3 += (0.5*rkstep*k2)
                    k3[:, 1:3] = y3[:, 4:6]
                    tmp = zeros(size(y3, 1))
                    sumabs2!(tmp, y3[:, 1:3])
                    tmp .*= sqrt(sqrt(tmp))
                    tmp .*= tmp
                    k3[:, 4:6] = y3[:, 1:3]
                    k3[:, 4:6] ./= tmp
                    k3[:, 4:6] .*= h2
                    k3[:, 4:6] .*= -1.5

                    y4 = copy(y)
                    y4 += (rkstep*k3)
                    k4[:, 1:3] = y4[:, 4:6]
                    tmp = zeros(size(y4, 1))
                    sumabs2!(tmp, y4[:, 1:3])
                    tmp .*= sqrt(sqrt(tmp))
                    tmp .*= tmp
                    k4[:, 4:6] = y4[:, 1:3]
                    k4[:, 4:6] ./= tmp
                    k4[:, 4:6] .*= h2
                    k4[:, 4:6] .*= -1.5

                    k2 *= 2.0
                    k3 *= 2.0
                    k1 += k2
                    k1 += k3
                    k1 += k4
                    k1 *= rkstep/6.0
                    velocity += k1[:, 4:6]
                end

                point += k1[:, 1:3]
            end

            pointsqr = sqrnorm(point)

            if (FOGDO != 0) && (it%FOGSKIP == 0)
                phsphtaper = clamp(0.8*(pointsqr-1.0), 0.0, 1.0)
                fogint = clamp(FOGMULT * FOGSKIP * STEP ./ pointsqr, 0.0, 1.0) .* phsphtaper
                fogcol = tmp_ones3

                object_colour = blendcolors(fogcol, fogint, object_colour, object_alpha)
                object_alpha = blendalpha(fogint, object_alpha)
            end

            if DISK_TEXTURE_INT != DT_NONE
                mask_crossing = (oldpoint[:, 2] .> 0.0) $ (point[:, 2] .> 0.0)
                mask_distance = (pointsqr .< DISKOUTERSQR) & (pointsqr .> DISKINNERSQR)
                diskmask = mask_crossing & mask_distance

                if any(diskmask)
                    lambdaa = -point[:, 2] ./ velocity[:, 2]
                    colpoint = point + reshape(lambdaa, size(lambdaa, 1), 1) .* velocity
                    colpointsqr = sqrnorm(colpoint)

                    if DISK_TEXTURE_INT == DT_GRID
                        phi = atan2(colpoint[:, 1], point[:, 3])
                        theta = atan2(colpoint[:, 2], vnorm(point[:, [1, 3]]))
                        tmp = mod(phi, 0.52359) .< 0.261799
                        tmp2 = [1.0, 1.0, 0.0]
                        tmp3 = [0.0, 0.0, 1.0]
                        diskcolor = tmp * tmp2' + tmp_ones * tmp3'
                        diskalpha = diskmask
                    elseif DISK_TEXTURE_INT == DT_SOLID
                        diskcolor = [1.0, 1.0, 0.98]
                        diskalpha = diskmask
                    elseif DISK_TEXTURE_INT == DT_TEXTURE
                        phi = atan2(colpoint[:, 1], point[:, 3])
                        uv = zeros(numChunk, 2)

                        uv[:, 1] = ((phi+2.0*pi) % (2.0*pi)) / (2.0*pi)
                        uv[:, 2] = (sqrt(colpointsqr) - DISKINNER) / (DISKOUTER - DISKINNER)

                        diskcolor = lookup(texarr_disk, clamp(uv, 0.0, 1.0))
                        diskalpha = diskmask .* clamp(sqrnorm(diskcolor)/3.0, 0.0, 1.0)
                    elseif DISK_TEXTURE_INT == DT_BLACKBODY
                        temperature = exp(disktemp(colpointsqr, 9.2103))

                        if REDSHIFT != 0
                            R = sqrt(colpointsqr)
                            tmp = (clamp(sqrt(colpointsqr)-1.0, 0.1, Inf))^(-0.5)
                            disc_velocity = 0.70710678 * reshape(tmp, size(tmp, 1), 1) .* cross(UPFIELD, normalize(colpoint))

                            gamma = (clamp(1.0-sqrnorm(disc_velocity), -Inf, 0.99))^(-0.5)

                            opz_doppler = gamma .* (1.0 + diag(disc_velocity * normalize(velocity)'))
                            opz_gravitational = (clamp(1.0 - 1.0 ./ R, 1.0, Inf))^(-0.5)

                            temperature ./= clamp(opz_doppler .* opz_gravitational, 0.1, Inf)
                        end

                        intensity_out = intensity(temperature)
                        if DISK_INTENSITY_DO != 0
                            tmp_M = colour(temperature)
                            tmp_x = DISK_MULTIPLIER * intensity_out
                            diskcolor = zeros(tmp_M)
                            for tmp_i=1:size(tmp_M, 1)
                                for tmp_j=1:size(tmp_M, 2)
                                    diskcolor[tmp_i, tmp_j] = tmp_M[tmp_i, tmp_j] * tmp_x[i]
                                end
                            end
                        else
                            diskcolor = colour(temperature)
                        end

                        iscotaper = clamp((colpointsqr-DISKINNERSQR)*0.3, 0.0, 1.0)
                        outertaper = clamp(temperature / 1000.0, 0.0, 1.0)

                        diskalpha = diskmask .* iscotaper .* outertaper
                    end

                    object_colour = blendcolors(diskcolor, diskalpha, object_colour, object_alpha)
                    object_alpha = blendalpha(diskalpha, object_alpha)
                end
            end

            oldpointsqr = sqrnorm(oldpoint)
            mask_horizon = (pointsqr .< 1.0) & (sqrnorm(oldpoint) .> 1.0)

            if any(mask_horizon)
                tmp = (1.0-oldpointsqr) ./ (pointsqr - oldpointsqr)
                lambdaa = 1.0 - reshape(tmp, size(tmp, 1), 1)
                colpoint = lambdaa .* point + (1.0-lambdaa) .* oldpoint

                if HORIZON_GRID != 0
                    phi = atan2(colpoint[:, 1], point[:, 3])
                    theta = atan2(colpoint[:, 2], vnorm(point[:, [1, 3]]))
                    u = (mod(phi, 1.04719) .< 0.52359) $ (mod(theta, 1.04719) .< 0.52359)
                    v = [1.0, 0.0, 0.0]
                    horizoncolour = u * v'
                else
                    horizoncolour = BLACK
                end

                horizonalpha = mask_horizon

                object_colour = blendcolors(horizoncolour, horizonalpha, object_colour, object_alpha)
                object_alpha = blendalpha(horizonalpha, object_alpha)
            end
        end

        vphi = atan2(velocity[:, 1], velocity[:, 3])
        vtheta = atan2(velocity[:, 2], vnorm(velocity[:, [1, 3]]))
        vuv = zeros(numChunk, 2)

        vuv[:, 1] = mod(vphi+4.5, 2.0*pi) / (2.0*pi)
        vuv[:, 2] = (vtheta+pi/2.0) / pi

        if SKY_TEXTURE_INT == DT_TEXTURE
            col_sky = lookup(texarr_sky, vuv)[:, 1:3]
        end

        if SKY_TEXTURE_INT == ST_TEXTURE
            col_bg = col_sky
        elseif SKY_TEXTURE_INT == ST_NONE
            col_bg = zeros(numChunk, 3)
        elseif SKY_TEXTURE_INT == ST_FINAL
            tmp = [0.5, 0.5, 0.0]
            dbg_finvec = clamp(normalize(velocity) + reshape(tmp, 1, size(tmp, 1)), 0.0, 1.0)
            col_bg = dbg_finvec
        else
            col_bg = zeros(numChunk, 3)
        end

        col_bg_and_obj = blendcolors(SKYDISK_RATIO*col_bg, tmp_ones, object_colour, object_alpha)

        if DISABLE_SHUFFLING != 0
            total_colour_buffer_preproc[chunk[1]:chunk[end], :] = col_bg_and_obj
        else
            total_colour_buffer_preproc[chunk, :] = col_bg_and_obj
        end
    end
end

@sync begin
    for (i, wpid) in enumerate(workers())
        @async begin
            remotecall_wait(wpid, raytrace_schedule, i, schedules[i], total_colour_buffer_preproc_shared)
        end
    end
end

tm = toc()
println("Computations took ", tm, " seconds")

println("Postprocessing")

total_colour_buffer_preproc_shared *= GAIN

if AIRY_BLOOM != 0
    colour_bloomd = copy(total_colour_buffer_preproc_shared)
    colour_bloomd = reshape(colour_bloomd, RESOLUTION[1], RESOLUTION[2], 3)

    radd = 0.00019825 * RESOLUTION[1] / atan(TANFOV)
    radd *= AIRY_RADIUS

    mxint = maximum(colour_bloomd)

    kern_radius = 25.0 * (maximum(colour_bloomd)/5.0)^(1.0/3.0) * RESOLUTION[1]/1920.0

    colour_bloomd = airy_convolve(colour_bloomd, radd)

    colour_bloomd = reshape(colour_bloomd, numPixels, 3)

    colour_pb = colour_bloomd
else
    colour_pb = total_colour_buffer_preproc_shared
end

if BLURDO != 0
    blurd = copy(total_colour_buffer_preproc_shared)
    blurd = reshape(blurd, RESOLUTION[1], RESOLUTION[2], 3)

    for i=1:2
        sigma = trunc(Int, 0.05 * RESOLUTION[1])
        blurd = Images.imfilter_gaussian(blurd, [sigma, sigma, sigma])
    end

    blurd = reshape(blurd, numPixels, 3)
    colourt = colour_pb + 0.2 * blurd
else
    colourt = colour_pb
end

if NORMALIZE > 0
    colourt *= 1.0 / (NORMALIZE * maximum(colourt))
end

colourt = clamp(colourt, 0.0, 1.0)

saveToImg(colourt, "tests/out.png")
saveToImg(total_colour_buffer_preproc_shared, "tests/preproc.png")
if BLURDO != 0
    saveToImg(colour_pb, "tests/postbloom.png")
end
