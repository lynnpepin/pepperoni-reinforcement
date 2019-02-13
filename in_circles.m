function Yesorno = in_circles(c,Cir)
    for i=1:length(Cir)
       if c.index == Cir(i).index
           Yesorno = 1;
           return;
       end
    end
    Yesorno = 0;
    return;
end