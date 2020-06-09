module Le

mutable struct Tensor
end

function tensor
    output_ptr = ccall(
        (:le_tensor_new, :lible),
        Ptr{Tensor},
        (Csize_t,),
        n
    )
    if output_ptr == C_NULL
        throw(OutOfMemoryError())
    end
    return output_ptr
end

end
