function theta = theta_arround(cv)
    r = cv.radius;
    theta=0;
    for i=1:length(cv.neighbors)-1
        rj = cv.neighbors(i).radius;
        rk = cv.neighbors(i+1).radius;
        theta = acos( ( (r+rj)^2 + (r+rk)^2 - (rj+rk)^2 )/( 2*(r+rj)*(r+rk) ) ) + theta;
    end
end