




/******************************************************************************

Implementation of k nearest neighbor information estimator based on kd tree,
including Kozachenko-Leonenko entropy estimator and KSG mutual information estimator.

*******************************************************************************/

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <random>
#include <cmath>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
using namespace std;

struct TreeNode{
    double thresh;
    int dim=-1; //-1 means stop splitting.
    int point;
    TreeNode *left;
    TreeNode *right;
};


void output(vector<double> v){
    int N,i;
    N=v.size();
    for(i=0;i<N-1;i++){
        cout<<v[i]<<",";
    }
    cout<<v[N-1]<<' ';
}

TreeNode *construct(vector<vector<double>> &X, int dstart){
    /*
    Create kd-tree.
    */
    int N,i,d,mid_index;
    TreeNode *head=(struct TreeNode *)malloc(sizeof(struct TreeNode));
    
    //Note that it is crucial to avoid vector in the struct, such that the malloc function is valid.
    
    TreeNode *left,*right;
    double median;
    vector<double> val;
    vector<vector<double>> Xleft,Xright;
    int dnext;
    N=X.size();
    d=X[0].size()-1; //Here -1 means that we need to remove the last item when calculating the dimension.
    head->point=-1; //starts with no point.
    if(N==0){
        cout<<"No data."<<endl;
        return head;
    } 
    if(N>1){
        vector<double> ind;
        for(i=0;i<N;i++){
            val.push_back(X[i][dstart]);
            ind.push_back(X[i][d]);
        }

        sort(val.begin(),val.end());
        if(N%2==1){
            mid_index=int((N-1)/2);
            median=val[mid_index];
        }else{
            mid_index=int(N/2)-1;
            median=val[mid_index];
        }
        dnext=(dstart+1)%d;
        for(i=0;i<N;i++){
            if(X[i][dstart]<median){
                Xleft.push_back(X[i]);  
            }else if (X[i][dstart]>median){
                Xright.push_back(X[i]);
            }
            else{
                if(head->point==-1){
                    //Here, X[i][d] is the index of the element X[i] in original dataset.
                    head->point=int(X[i][d]);
                }else{
                    Xleft.push_back(X[i]);
                }
            }             
        }
        //if(Xleft.size()==1) {cout<<"test:";output(Xleft[0]);cout<<endl;}
        /*
        if(N==9){
            cout<<"First splitting:"<<endl;
            int n1,n2;
            n1=Xleft.size();
            n2=Xright.size();
            cout<<n1<<' '<<n2<<endl;
            for(i=0;i<n1;i++) output(Xleft[i]);cout<<endl;
            for(i=0;i<n2;i++) output(Xright[i]);cout<<endl;
        }
        */
 
        if(Xleft.empty()==0){
            head->left=construct(Xleft,dnext);
        }else{
            head->left=NULL;
        }
        
        if(Xright.empty()==0) {
            head->right=construct(Xright,dnext); 
        }else{
            head->right=NULL;
        }
        
        head->thresh=median;
        head->dim=dstart;
    }else{
        head->point=int(X[0][d]);
        head->dim=-1;
        head->left=NULL;
        head->right=NULL;
    }
    //if(head->left) cout<<"test:"<<head->left->point<<endl;
    return head;
}

TreeNode *kdtree(vector<vector<double>> &X){
    int N,i;
    TreeNode *root;
    N=X.size();
    for(i=0;i<N;i++){
        X[i].push_back(double(i)); //This step assigns each data point with an index.
    }
    /*
    The construction of kdtree starts at the first dimension, using the augmented
    data (i.e. each data point is assigned with an index.)
    */
    root=construct(X,0);
    
    //Now remove the indices in X.
    for(i=0;i<N;i++){
        X[i].pop_back();
    }
    
    return root;
}

double distance(vector<double> &u, vector<double> &v, int type){
    /*
    Calculate the distance between two vectors.
    type=1: Chebyshev distance (maximum distance).
    type=2: Eucledian distance.
    */
    int n1,n2,i;
    double d=0;
    vector<double> dif;
    n1=u.size();
    n2=v.size();
    if(n1!=n2){
        cout<<"The lengths of two vectors are not equal."<<endl;
    }
    for(i=0;i<n1;i++){
        dif.push_back(fabs(u[i]-v[i]));
    }
    switch(type){
        case 1:{
            for(i=0;i<n1;i++){
                if(dif[i]>d) d=dif[i];
            }
            return d;
        }
        case 2:{
            for(i=0;i<n1;i++){
                d+=dif[i]*dif[i];
            }
            return sqrt(d);
        }
    }
}

double dist_node_to_point(TreeNode *node, vector<double> v){
    int d=node->dim;
    return fabs(v[d]-node->thresh);
}

vector<double> knndist_compute(TreeNode *root, vector<double> v, int k, vector<double> distmin,
    vector<vector<double>> &X){
    /*
    Calculate k nearest neighbor distances.
    */
    vector<TreeNode *> L;
    vector<double> s; //s records the distance from point v to each separating hyperplanes.
    vector<bool> direction; //0:left, 1:right.
    int i,d,N,N_prev,M;
    TreeNode *p;
    double newdist;
    //output(distmin); //Debug: check whether nearest distances are correct.
    p=root;
    L.push_back(p);
    N=1;
    newdist=distance(v,X[root->point],2);
    if(distmin.front()>newdist){
        /*
        Construct a priority queue to store the k nearest neighbors using max heap.
        */
        pop_heap(distmin.begin(),distmin.end());
        distmin.pop_back();
        distmin.push_back(newdist);
        push_heap(distmin.begin(),distmin.end());
    }
    s.push_back(dist_node_to_point(root,v));
    while(p->dim!=-1){
        N_prev=N;
        d=p->dim;
        if(v[d]<=p->thresh && p->left){
            p=p->left;
            L.push_back(p);N++;
            s.push_back(dist_node_to_point(p,v));
            direction.push_back(0);
            newdist=distance(v,X[p->point],2);
            if(distmin.front()>newdist){
                pop_heap(distmin.begin(),distmin.end());
                distmin.pop_back();
                distmin.push_back(newdist);
                push_heap(distmin.begin(),distmin.end());
            }
        }else if (v[d]>p->thresh && p->right){
            p=p->right;
            L.push_back(p);N++;
            s.push_back(dist_node_to_point(p,v));
            direction.push_back(1);
            newdist=distance(v,X[p->point],2);
            if(distmin.front()>newdist){
                pop_heap(distmin.begin(),distmin.end());
                distmin.pop_back();
                distmin.push_back(newdist);
                push_heap(distmin.begin(),distmin.end());
            }
        }
        if(N==N_prev) break;
    }
    if(p->dim!=-1){
        // Record the direction of the last step.
        d=p->dim;
        if(v[d]<=p->thresh) direction.push_back(0);
        else direction.push_back(1);
    }
    
    M=direction.size();
    //cout<<"Size:"<<M<<endl;
    
    for(i=M-1;i>=0;i--){
        if(s[i]<distmin.front()){
            if(direction[i]==0 && L[i]->right){
                distmin=knndist_compute(L[i]->right,v,k,distmin,X);
            }else if(direction[i]==1 && L[i]->left){
                distmin=knndist_compute(L[i]->left,v,k,distmin,X);
            }
        }
    }
    return distmin;
}

double knndist(TreeNode *root, vector<double> v,int k, vector<vector<double>> &X){
    vector<double>distmin;
    double firstdist;
    int i;
    
    firstdist=distance(v,X[root->point],2);
    distmin.push_back(firstdist);
    for(i=1;i<k;i++){
        distmin.push_back(numeric_limits<double>::max());
    }
    /*
    The initial heap stores the distance from the query point to the first splitting 
    hyperplane, and (k-1) infinity values.
    */
    make_heap(begin(distmin),end(distmin));
    distmin=knndist_compute(root,v,k,distmin,X);
    return distmin.front();
}

double KL(vector<vector<double>> &X, int k){
    //Kozachenko-Leonenko entropy estimate.
    TreeNode *root;
    int i,N,d;
    double t;
    double s=0;
    double c_d;
    root=kdtree(X);
    N=X.size();
    d=X[0].size();
    
    c_d=pow(M_PI,double(d)/2)/tgamma(double(d)/2+1);
    for(i=0;i<N;i++){
        t=knndist(root,X[i],k+1,X);
        //cout<<t<<endl;
        s+=log(t);
    }
    
    s=s/N;
    //cout<<s<<endl;
    return -boost::math::digamma(k)+boost::math::digamma(N)+log(c_d)+double(d)*s;
}

void showtree(TreeNode *root, vector<vector<double>> &X){
    /*
    This subprogram displays the kdtree structures.
    */
    vector<TreeNode *> L;
    TreeNode *nullnode=(struct TreeNode *)malloc(sizeof(struct TreeNode));
    TreeNode *p;
    int i=0,d=0,M,T=1,t=0;
    p=root;
    L.push_back(p);
    nullnode->point=-1;
    //cout<<"check1"<<endl;
    while(L.size()>i){
        //cout<<i<<endl;
        if(L[i]->point!=-1){
            if(L[i]->left) {
                L.push_back(L[i]->left);
            }else{
                L.push_back(nullnode);
                //cout<<"null:"<<L.back()->point<<endl;
            }
            if(L[i]->right){
                L.push_back(L[i]->right);
            }else{
                L.push_back(nullnode);
            }
        }
        i++;
    }
    //cout<<"check2"<<endl;
    while(L.back()->point==-1){
        L.pop_back();
    }
    M=L.size();
    cout<<"kdtree:"<<endl;
    //cout<<L[2]->point.size()<<endl;
    //output(L[2]->point);
    for(i=0;i<M;i++){
        if(L[i]->point!=-1) output(X[L[i]->point]);
        else cout<<"null ";
        t++;
        if(t==T){
            T*=2;
            t=0;
            cout<<endl;
        }
    }
}

int main(){
    vector<vector<double> > X;
    vector<double> temp;
    vector<double> v1{0,0};
    vector<double> v2{0.3,0.4};
    int N=100,i,d=2,j;
    TreeNode *head,*p;
    double dist;
    double entropy;
    
    //Generate random variables with Gaussian distribution.
    
    default_random_engine generator;
    normal_distribution<double> distribution(0,1);
    //cout<<"Initial points:"<<endl;
    for(i=0;i<N;i++){
        vector<double> temp;
        for(j=0;j<d;j++){
            double number=distribution(generator);
            temp.push_back(number);
        }
        X.push_back(temp);
        //output(temp);cout<<endl;
    }
    //cout<<X.size()<<' '<<X[1].size()<<endl;
    //head=kdtree(X);
    //output(head->point);
    //output(head->left->point);
    //output(head->right->point);
    //output(head->left->right->point);
    //dist=knndist(head,v1,3,X);
    //cout<<dist<<endl;
    entropy=KL(X,3);
    cout<<entropy<<endl;
    return 0;
}





