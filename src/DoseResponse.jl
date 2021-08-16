module DoseResponse

    using LsqFit
    import LsqFit: LsqFitResult
    using StatsBase
    using GLM
    using Distributions
    using LinearAlgebra

    struct DoseResponseModel{Y <: AbstractVector, X <: AbstractVector, P,R,J,W <: AbstractArray} <: StatisticalModel 
        result::LsqFitResult{P,R,J,W}
        response::Y
        dose::X
    end

    function DoseResponseModel(response::AbstractVector{T}, dose::AbstractVector{T}, theta0::AbstractVector{T}) where T <: Real
        fit = curve_fit(generalized_logistic_5PL!, jacobian_5PL!, dose, response, theta0, inplace=true)
        DoseResponseModel(fit, response, dose)
    end
    function DoseResponseModel(response::AbstractVector{T}, dose::AbstractVector{T}) where T <: Real
        theta0 = zeros(eltype(response), 5)
        initial_values!(theta0, response, dose)
        fit = curve_fit(generalized_logistic_5PL!, jacobian_5PL!, dose, response, theta0, inplace=true)
        DoseResponseModel(fit, response, dose)
    end
    function initial_values!(theta0::AbstractVector{T}, response::AbstractVector{T}, dose::AbstractVector{T}) where T <: Real
        ymin, ymax = minimum(response), maximum(response)
        theta0[1] = ymin - 0.001 * (ymax-ymin)
        theta0[2] = ymax + 0.001 * (ymax-ymin)

        ytrans = log.((theta0[2] .- response) ./ (response .- theta0[1]))
        idx = findall(dose .> 0)
        m = lm([ones(length(idx)) log.(x[idx])], ytrans[idx])
        theta0[3] = -coef(m)[2]
        theta0[4] = -coef(m)[1]/(-theta0[3])
        theta0[5] = 0.0
    end

    StatsBase.coef(m::DoseResponseModel) = m.result.param
    StatsBase.coefnames(m::DoseResponseModel) = ["A","K","B","M","logv"]
    StatsBase.confint(m::DoseResponseModel) = confidence_interval(m.result)
    StatsBase.dof(m::DoseResponseModel) = length(m.result.param) + 1
    StatsBase.nobs(m::DoseResponseModel) = length(m.result.resid)
    StatsBase.rss(m::DoseResponseModel) = sum(abs2, m.result.resid)
    StatsBase.weights(m::DoseResponseModel) = m.result.wt
    StatsBase.residuals(m::DoseResponseModel) = m.result.resid
    StatsBase.dof_residual(m::DoseResponseModel) = nobs(m) - length(coef(m))
    mse(m::DoseResponseModel) = rss(m)/dof_residual(m)

    function StatsBase.stderror(m::DoseResponseModel; rtol::Real=NaN, atol::Real=0)
        # computes standard error of estimates from
        #   fit   : a LsqFitResult from a curve_fit()
        #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
        #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
        covar = LsqFit.estimate_covar(m.result)
        # then the standard errors are given by the sqrt of the diagonal
        vars = diag(covar)
        vratio = minimum(vars)/maximum(vars)
        if !isapprox(vratio, 0.0, atol=atol, rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol) && vratio < 0.0
            error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
        end
        return sqrt.(abs.(vars))
    end
    
    function StatsBase.coeftable(m::DoseResponseModel; level::T=0.95) where T <: Real
        cc = coef(m)
        se = stderror(m)
        tt = cc ./ se
        p = ccdf.(Ref(FDist(1, dof_residual(m))), abs2.(tt))
        ci = se*quantile(TDist(dof_residual(m)), (1-level)/2)
        levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
        CoefTable(hcat(cc,se,tt,p,cc+ci,cc-ci),
                ["Coef.","Std. Error","t","Pr(>|t|)","Lower $levstr%","Upper $levstr%"],
                coefnames(m), 4, 3)
    end

    StatsBase.loglikelihood(m::DoseResponseModel) = -nobs(m)/2 * (log(2π) + log(rss(m)) - log(nobs(m))+1)

    StatsBase.predict(m::DoseResponseModel) = generalized_logistic_5PL(m.dose, m.result.param)
    StatsBase.predict(m::DoseResponseModel{Y,X,P,R,J,W}, newX::X) where {Y<:AbstractVector,X<:AbstractVector,P,R,J,W<:AbstractArray} = generalized_logistic_5PL(newX, m.result.param)
    StatsBase.mss(m::DoseResponseModel) = sum(abs2, predict(m) .- mean(m.response))
    StatsBase.r2(m::DoseResponseModel) = 1 - rss(m)/sum(abs2,m.response .- mean(m.response))
    StatsBase.islinear(m::DoseResponseModel) = false




    ## 4PL
    function generalized_logistic_4PL!(F, x,theta)
        F .= ifelse.(isfinite.(log.(x)), theta[1] .+ (theta[2] - theta[1]) ./ (1.0 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))), theta[1] + (theta[2] - theta[1]) * ((theta[3] < 0) && ((theta[2] - theta[1]) > 0)))
    end
    function generalized_logistic_4PL(x,theta)
        ifelse.(isfinite.(log.(x)), theta[1] .+ (theta[2] - theta[1]) ./ (1.0 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))), theta[1] + (theta[2] - theta[1]) * ((theta[3] < 0) && ((theta[2] - theta[1]) > 0)))
    end

    function jacobian_4PL!(J, x, theta)
        J[:,1] .= 1 .- (1 .+ exp.(-theta[3] .* (log.(x) .- theta[4])))
        J[:,2] .= (1 .+ exp.(-theta[3] .* (log.(x) .- theta[4])))
        J[:,3] .= ifelse.(isfinite.(log.(x)), (theta[2] - theta[1]) .* (log.(x) .- theta[4]) .* exp.(-theta[3] .* (log.(x) .- theta[4])) ./ (1 .+ exp.(-theta[3] .* (log.(x) .- theta[4])) ), 0.0)
        J[:,4] .=  -(theta[2] - theta[1]) .* theta[3] .* exp.(-theta[3] .* (log.(x) .- theta[4])) ./ (1 .+ exp.(-theta[3] .* (log.(x) .- theta[4])) )
    end

    ## 5PL 
    function generalized_logistic_5PL!(F, x, theta)
        F .= ifelse.(isfinite.(log.(x)), theta[1] .+ (theta[2] - theta[1]) ./ (1.0 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))) .^ (1 / exp(theta[5])), theta[1] + (theta[2] - theta[1]) * ((theta[3] < 0) && ((theta[2] - theta[1]) > 0)))
    end
    function generalized_logistic_5PL(x, theta)
        ifelse.(isfinite.(log.(x)), theta[1] .+ (theta[2] - theta[1]) ./ (1.0 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))) .^ (1 / exp(theta[5])), theta[1] + (theta[2] - theta[1]) * ((theta[3] < 0) && ((theta[2] - theta[1]) > 0)))
    end

    function jacobian_5PL!(J, x, theta)
        J[:,1] .= 1 .- (1 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))) .^ (-1 / exp(theta[5]))
        J[:,2] .= (1 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))) .^ (-1 / exp(theta[5]))
        J[:,3] .= ifelse.(isfinite.(log.(x)), (theta[2] - theta[1]) .* (log.(x) .- theta[4]) .* exp.(-theta[3] .* (log.(x) .- theta[4])) ./ (exp(theta[5]) .* (1 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))) .^ (1/exp(theta[5]) + 1)), 0.0)
        J[:,4] .=  -(theta[2] - theta[1]) .* theta[3] .* exp.(-theta[3] .* (log.(x) .- theta[4])) ./ (exp(theta[5]) .* (1 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))) .^ (1/exp(theta[5]) + 1))
        J[:,5] .= (theta[2] - theta[1]) .* log.(1 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))) ./ (exp(theta[5]) .*(1 .+ exp.(-theta[3] .* (log.(x) .- theta[4]))) .^ (1/exp(theta[5])))
    end




end # module