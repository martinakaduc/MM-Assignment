from policy import Policy
import numpy as np
class Policy2310763(Policy):
    def __init__(self, ma_chinh_sach=1):
        # Đảm bảo mã chính sách hợp lệ
        if ma_chinh_sach == 1:
            self.policy_id = 1
        elif ma_chinh_sach == 2:
            self.policy_id = 2
        
    
    def get_action(self, quan_sat, thong_tin):
    # Kiểm tra chính sách và thực hiện theo từng chính sách
        if self.policy_id == 1:
            san_pham = sorted(quan_sat["products"], key=lambda sp: sp["size"][0] * sp["size"][1], reverse=True)
            kho_sap_xep = sorted(enumerate(quan_sat["stocks"]), key=lambda k: self._get_stock_size_(k[1])[0] * self._get_stock_size_(k[1])[1])

            for sp in san_pham:
                if sp["quantity"] == 0:
                    continue

                rong_sp, cao_sp = sp["size"]

                for idx, kho in kho_sap_xep:
                    rong_kho, cao_kho = self._get_stock_size_(kho)

                    if np.all(kho == -1):  # Kho trống, đặt vào góc trên bên trái
                        return {
                            "stock_idx": idx,
                            "size": (rong_sp, cao_sp),
                            "position": (0, 0),
                        }

                    vi_tri_tot_nhat = self._tim_vi_tri_tot_nhat_(kho, rong_kho, cao_kho, (rong_sp, cao_sp), idx)
                    if vi_tri_tot_nhat:
                        return vi_tri_tot_nhat

            # Không tìm được vị trí hợp lệ
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        elif self.policy_id == 2:
            san_pham = sorted(quan_sat["products"], key=lambda sp: sp["size"][0] * sp["size"][1], reverse=True)

            for sp in san_pham:
                if sp["quantity"] > 0:
                    ket_qua = self._tim_vi_tri_(quan_sat, sp)
                    if ket_qua:
                        kho_idx, vi_tri_x, vi_tri_y = ket_qua
                        return {
                            "stock_idx": kho_idx,
                            "size": sp["size"],
                            "position": (vi_tri_x, vi_tri_y),
                        }

            # Không tìm được vị trí hợp lệ
            return None

        # Nếu chính sách không hợp lệ
        return None

    def _tim_vi_tri_tot_nhat_(self, kho, rong_kho, cao_kho, kich_thuoc_sp, idx):
        rong_sp, cao_sp = kich_thuoc_sp
        vi_tri_tot_nhat = None
        luong_lang_phi_nho_nhat = float('inf')

        for huong_dat in [(rong_sp, cao_sp), (cao_sp, rong_sp)]:
            max_x = rong_kho - huong_dat[0]
            max_y = cao_kho - huong_dat[1]

            for x in range(max_x + 1):
                for y in range(max_y + 1):
                    if self._can_place_(kho, (x, y), huong_dat):
                        lang_phi = self._tinh_lang_phi_(kho, huong_dat)

                        if lang_phi < luong_lang_phi_nho_nhat:
                            luong_lang_phi_nho_nhat = lang_phi
                            vi_tri_tot_nhat = {
                                "stock_idx": idx,
                                "size": huong_dat,
                                "position": (x, y),
                            }

        return vi_tri_tot_nhat

    def _tinh_lang_phi_(self, kho, kich_thuoc_sp):
        dien_tich_su_dung = kich_thuoc_sp[0] * kich_thuoc_sp[1]
        tong_dien_tich_trong = np.sum(kho == -1)
        return tong_dien_tich_trong - dien_tich_su_dung

    def _tim_vi_tri_(self, quan_sat, sp):
        rong_sp, cao_sp = sp["size"]

        for idx, kho in enumerate(quan_sat["stocks"]):
            rong_kho, cao_kho = self._get_stock_size_(kho)

            if rong_kho < rong_sp or cao_kho < cao_sp:
                continue

            for x in range(rong_kho - rong_sp + 1):
                for y in range(cao_kho - cao_sp + 1):
                    if self._can_place_(kho, (x, y), sp["size"]):
                        return idx, x, y

        return None
